import torch

c1 = 16
candidates_num = c1 * (c1 - 1) // 2
can_idx_loc = torch.zeros((3, candidates_num * 2), dtype=torch.int64)
can_idx_values = torch.ones(candidates_num * 2)
can_idx = 0
for i in range(c1):
    for j in range(i + 1, c1):
        can_idx_loc[:, can_idx * 2] = torch.tensor([0, can_idx, i])
        can_idx_loc[:, can_idx * 2 + 1] = torch.tensor([1, can_idx, j])
        can_idx += 1
candidates_met = torch.sparse_coo_tensor(indices=can_idx_loc, values=can_idx_values)
candidates_met = candidates_met.to_dense()
aa = torch.argmax(candidates_met, dim=2)

loss_fn = torch.nn.CrossEntropyLoss()
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