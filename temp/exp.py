import torch
from torch import Tensor, nn
    
    
class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, source: Tensor, transformed: Tensor):
        ctx.save_for_backward(source)
        return transformed

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        print("Backward called!")  # 调试信息
        source, = ctx.saved_tensors
        grad_input = grad_output.clone()  # 直接回传梯度
        return grad_input, None


class HadamardExpansion(nn.Module):
    def __init__(self, c1, ce):
        super().__init__()

        self.c1 = c1
        self.ce = ce
        self.candis_num = c1 * (c1 - 1) // 2
        assert self.ce <= self.candis_num, f"too much expansion channels required"
        
        # hadamard-pairs selected
        self.selected_met: Tensor
        self.selected_seq: Tensor
        # self.register_buffer('selected_met', torch.zeros((2, self.ce, self.c1)))
        # self.register_buffer('selected_seq', torch.zeros((2, self.ce), dtype=torch.int64))
        
        # hadamard-pairs candidates
        can_idx_loc = torch.zeros((3, self.candis_num * 2), dtype=torch.int64)
        can_idx_val = torch.ones(self.candis_num * 2)
        can_idx = 0
        for i in range(c1):
            for j in range(i + 1, c1):
                can_idx_loc[:, can_idx * 2] = torch.tensor([0, can_idx, i])
                can_idx_loc[:, can_idx * 2 + 1] = torch.tensor([1, can_idx, j])
                can_idx += 1
        self.candis_met: Tensor
        self.register_buffer('candis_met', torch.sparse_coo_tensor(indices=can_idx_loc, values=can_idx_val).to_dense())

        # gumbel-softmax
        self.logits = nn.Parameter(torch.randn(self.candis_num))
        self.tau = nn.Parameter(torch.tensor(2.0))
        self.candis_to_ce: Tensor
        self.register_buffer('candis_to_ce', torch.zeros(self.ce, self.candis_num))

        # instance normalize
        self.norm = nn.InstanceNorm2d(c1 + ce, affine=True)
        
        # initialize
        torch.nn.init.uniform_(self.logits, a=-0.1, b=0.1)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            self._update_mask()
            return self.selected_met
            _shape = list(x.shape)
            _shape[1] = -1
            x = x.flatten(2)
            x_i = (self.selected_met[0, ...] @ x).view(_shape)
            x_j = (self.selected_met[1, ...] @ x).view(_shape)
            x = x.view(_shape)
        else:
            x_i = x[:, self.selected_seq[0], ...]
            x_j = x[:, self.selected_seq[1], ...]
        x_expand = x_i * x_j
        exp = self.logits.grad
        return self.norm(torch.cat([x, x_expand], dim=1))
        
    def _update_mask(self):
        self.tau.data = torch.clamp(self.tau, max=4.0, min=0.1)   
        mask = nn.functional.gumbel_softmax(self.logits, tau=self.tau)
        
        # mask, topk_idx = Functional.gumbel_topk(self.logits, self.ce, tau=self.tau, hard=True)
        
        _, topk_idx = torch.topk(mask, self.ce)
        # hard_mask_ = torch.zeros_like(self.logits).scatter_(0, topk_idx, 1.0)
        # hard_mask = StraightThroughEstimator.apply(mask, hard_mask_)
        # hard_mask = hard_mask_ + self.logits.detach() - self.logits
        
        self.candis_to_ce[torch.arange(self.ce), topk_idx] = 1.0
        self.candis_to_ce *= mask.unsqueeze(0)

        # selected channel
        self.selected_met = self.candis_to_ce @ self.candis_met
        self.selected_seq = torch.argmax(self.selected_met, dim=2)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tau = torch.nn.Parameter(torch.randn(1))
        self.logits = torch.nn.Parameter(torch.randn(3))
        self.candis: Tensor
        self.register_buffer('candis', torch.tensor([[0.2, 0.6, 0.2]]))
        self.candis_to_ce: Tensor
        self.register_buffer('candis_to_ce', torch.zeros(2, 3))
        self.he = HadamardExpansion(3, 2)
        
    def forward(self):
        self.tau.data = torch.clamp(self.tau, max=4.0, min=0.1)        
        mask = torch.nn.functional.gumbel_softmax(self.logits, tau=self.tau)
        _, topk_idx = torch.topk(mask, 2)
        hard_mask = torch.zeros_like(self.logits).scatter_(0, topk_idx, 1.0)
        ret = StraightThroughEstimator.apply(mask, hard_mask)
        self.candis_to_ce[torch.arange(2), topk_idx] = 1.0
        res = self.candis_to_ce * ret.unsqueeze(0)
        a = res @ self.candis.T
        # a = torch.stack((a,a,a), dim=0)
        # return self.he(a.view(1, 3, 2, 1))
        return a
    
model = Model()
# opt = torch.optim.AdamW(model.parameters(), lr=0.01)

while True:
    output = model()
    # opt.zero_grad()
    loss = output.sum()
    loss.backward()
    # opt.step()

# 计算损失并反向传播
loss = output.sum()
loss.backward(retain_graph=True)  # 保留计算图

# 如果需要再次反向传播
another_loss = output.sum()
another_loss.backward()  # 此时计算图仍存在

exit()

import torch
from torch import nn, Tensor

loss_fn = torch.nn.CrossEntropyLoss()

c1 = 5
ce = 2
candidates_num = c1 * (c1 - 1) // 2
logits = nn.Parameter(torch.randn(candidates_num))
mask = nn.functional.gumbel_softmax(logits, tau=2.0)

can_idx_loc = torch.zeros((3, candidates_num * 2), dtype=torch.int64)
can_idx_values = torch.ones(candidates_num * 2)
can_idx = 0
for i in range(c1):
    for j in range(i + 1, c1):
        can_idx_loc[:, can_idx * 2] = torch.tensor([0, can_idx, i])
        can_idx_loc[:, can_idx * 2 + 1] = torch.tensor([1, can_idx, j])
        can_idx += 1
candidates_met = torch.sparse_coo_tensor(indices=can_idx_loc, values=can_idx_values)

_, topk_idx = torch.topk(mask, ce)

hard_mask = torch.zeros(candidates_num)
hard_mask_exp = torch.zeros(ce, candidates_num)
hard_mask[topk_idx] = 1.0

# straight-through estimator
hard_mask = hard_mask - mask.detach() + mask

# slice selected channel
ones_indices = (hard_mask == 1).nonzero(as_tuple=True)[0]
hard_mask_exp[torch.arange(ce), ones_indices] = 1.0
filt = hard_mask_exp * hard_mask.unsqueeze(0)
selected_met = filt @ candidates_met.to_dense()
selected_seq = torch.argmax(selected_met, dim=2)
loss = selected_met.sum()
print(logits.grad)
loss.backward()
print(logits.grad)

label = torch.tensor([2], dtype=torch.int64)
I = torch.eye(5)
data = torch.randn([1, 5, 4, 4])
avgpool = torch.nn.AdaptiveAvgPool2d(1)
net = torch.nn.Sequential(
    torch.nn.Linear(2, 6),
    torch.nn.ReLU6(inplace=True),
    torch.nn.Linear(6, 5)
)

logits = torch.randn((1, 5), requires_grad=True)
logits_non = torch.randn(5, requires_grad=True)

mask = torch.nn.functional.gumbel_softmax(logits, tau=1.0, hard=False)
_, topk_idx = torch.topk(mask, 2)
hard_mask = torch.zeros_like(mask)
hard_mask[:, topk_idx] = 1.0
hard_mask = hard_mask - mask.detach() + mask

selected_met = hard_mask * I
non_zero = torch.sum(selected_met, dim=1) != 0
selected_met = selected_met[non_zero]

x: torch.Tensor = selected_met @ data.view(1, 5, -1)
x = avgpool(x.view(1, 2, 4, 4))
x = net(x.flatten(1))

loss = loss_fn(x, label)
loss.backward()

print(mask)
print(loss)
print(logits.grad)  # 输出非零梯度
print(logits_non.grad)  # 输出非零梯度