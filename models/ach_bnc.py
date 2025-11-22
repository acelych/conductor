import math
import torch
from torch import nn, Tensor

class ECA(nn.Module):
    """ECA module for BNC format (e.g., ViT patch embeddings).
    Args:
        k_size: Kernel size for 1D conv on channel dimension.
        weights_only: If True, returns attention weights (before sigmoid), shape [B, 1, C].
    """
    def __init__(self, k_size=3, weights_only=False):
        super().__init__()
        self.weights_only = weights_only
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        if not weights_only:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: input features with shape [B, N, C]
        Returns:
            If weights_only: [B, 1, C] (raw conv output, no sigmoid)
            Else: [B, 1, C] (sigmoid-scaled channel attention weights)
        """
        # Global average pooling over N (tokens), keepdims for broadcasting
        # y: [B, 1, C]
        y = x.mean(dim=1, keepdim=True)  # equivalent to GAP in spatial domain

        # Transpose to [B, C, 1] -> unsqueeze to [B, C, 1] is not needed;
        # Instead: reshape y to [B, 1, C] → then permute to [B, C, 1]? Wait — conv1d expects [B, in_channels, L]
        # We want to apply 1D conv along C (channel dim), so treat C as "sequence length"

        # y: [B, 1, C] → transpose to [B, C, 1] is WRONG.
        # Correct: treat C as "temporal" dim → input to Conv1d must be [B, 1, C]
        # So y is already [B, 1, C] — perfect for Conv1d(1, 1, ...)

        y = self.conv(y)  # [B, 1, C] → conv along C-dim

        if not self.weights_only:
            y = self.sigmoid(y)

        return y  # [B, 1, C]

    # Optional: if you want to apply attention directly in forward (not just return weights)
    def apply_attention(self, x: Tensor) -> Tensor:
        """
        Apply ECA attention to x (B, N, C) → returns scaled x.
        """
        weights = self.forward(x)  # [B, 1, C]
        return x * weights  # broadcasting: [B, N, C] * [B, 1, C] → [B, N, C]

class DySoft(nn.Module):
    def __init__(self, dim, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(1, 1, dim))
        self.bias = nn.Parameter(torch.zeros(1, 1, dim))
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.alpha * x
        x = x / (1 + torch.abs(x))
        return x * self.weight + self.bias

class AdaptiveCrossHadamard(nn.Module):
    def __init__(self, c1, cs, norm: nn.Module, 
                 tau_init = 4.0,
                 tau_alpha = 0.005,
                 tau_clamp_max = 4.0,
                 tau_clamp_min = 0.01,
                 ):
        super().__init__()
        
        self.tau_init = tau_init
        self.tau_alpha = tau_alpha
        self.tau_clamp_max = tau_clamp_max
        self.tau_clamp_min = tau_clamp_min

        self.c1 = c1
        self.cs = cs
        self.cs_expand = cs * (cs - 1) // 2
        self.ce = c1 + self.cs_expand
        
        # fc-expand: 1x1 conv → Linear in BNC
        self.fc = nn.Linear(c1, c1)  # operates on last dim (C)
        # self.norm_x = nn.LayerNorm(c1)  # norm over channel dim (C), shape [B, N, C]
        
        # eva-net: use BNC version of ECA
        self.eva_net = ECA(k_size=5, weights_only=True)  # returns [B, 1, C]

        # gumbel-softmax
        self.tau = nn.Parameter(torch.tensor(self.tau_init), requires_grad=False)
        self.tau_adj = nn.Parameter(torch.tensor(0.0), requires_grad=False)  # dtype float
        
        # cross-hadamard indices (unchanged)
        self.hadamard_i: Tensor
        self.hadamard_j: Tensor
        self.register_buffer("hadamard_i", torch.zeros(self.cs_expand, dtype=torch.int64))
        self.register_buffer("hadamard_j", torch.zeros(self.cs_expand, dtype=torch.int64))
        h_idx = 0
        for i in range(self.cs):
            for j in range(i + 1, self.cs):
                self.hadamard_i[h_idx] = i
                self.hadamard_j[h_idx] = j
                h_idx += 1

        # expand normalize: same usage — norm([B, N, D]) where D = cs_expand
        self.norm = norm(self.ce)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, N, C] with C == c1
        x = self.fc(x)       # [B, N, C]
        return x
        # x = self.norm_x(x)   # [B, N, C]
        
        x_sel_ex = self._get_selected(x)  # [B, N, cs_expand]
        # x_sel_ex = self.norm(x_sel_ex)    # [B, N, cs_expand]
        x_out = torch.cat([x, x_sel_ex], dim=-1)
        
        return self.norm(x_out)  # [B, N, c1 + cs_expand]

    def _get_selected(self, x: Tensor) -> Tensor:
        # x: [B, N, C]
        # eva_net expects [B, N, C] → returns [B, 1, C]
        logits = self.eva_net(x).squeeze(1)  # [B, C], equivalent to flatten in BCHW case
        
        if self.training:
            B, N, C = x.shape
            
            mask = nn.functional.gumbel_softmax(logits, tau=self.tau, hard=False)  # [B, C]
            self._adjust_tau_with_grad(self.eva_net.conv.weight.grad)
            
            # Get top-k channel indices
            _, topk_idx = torch.topk(mask, self.cs, dim=1)  # [B, cs]
            
            # —— Exactly same as original ——
            _batchs = torch.arange(B).unsqueeze(-1).expand_as(topk_idx).flatten(0)
            _rows = torch.arange(self.cs).unsqueeze(0).expand_as(topk_idx).flatten(0)
            
            hard_mask_ = torch.zeros_like(mask)
            hard_mask_[_batchs, topk_idx.flatten()] = 1.0
            hard_mask = hard_mask_ + mask.detach() - mask

            mask_mat_ = torch.zeros(B, self.cs, C, device=x.device, dtype=x.dtype)
            mask_mat_[_batchs, _rows, topk_idx.flatten()] = 1.0
            mask_mat = mask_mat_ * hard_mask.unsqueeze(1)
            # ————————————————

            # Only change: x → x.transpose(1, 2)
            x_sel_flat = torch.bmm(mask_mat, x.transpose(1, 2))  # [B, cs, N]
            x_sel = x_sel_flat.transpose(1, 2)                   # [B, N, cs]

            return x_sel[:, :, self.hadamard_i] * x_sel[:, :, self.hadamard_j]

        else:
            _, topk_idx = torch.topk(logits, self.cs, dim=1)  # [B, cs]
            x_sel = torch.gather(x, dim=2, index=topk_idx.unsqueeze(1).expand(-1, x.size(1), -1))
            x_i = x_sel[:, :, self.hadamard_i]
            x_j = x_sel[:, :, self.hadamard_j]
            x_sel_ex = x_i * x_j

        return x_sel_ex

    def _adjust_tau_with_grad(self, grad: Tensor):
        if self.tau_adj.data != 0 and grad is not None:
            # Ensure tau_adj is float
            grad_norm = grad.norm()
            alpha = self.tau_alpha
            alpha *= 1.0 if self.tau_adj <= grad_norm else -1.0
            self.tau.data = torch.clamp(self.tau * (1.0 + alpha), max=self.tau_clamp_max, min=self.tau_clamp_min)
        if grad is not None:
            self.tau_adj.data = grad.norm()
            

class AdaptiveBottleneckBNC(nn.Module):
    def __init__(self, in_channel: int, ex_channel: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ci = in_channel
        self.ach = AdaptiveCrossHadamard(in_channel, ex_channel, DySoft)
        self.ce = self.ach.ce
        self.act = nn.SiLU()
        self.prj = nn.Linear(self.ci, self.ci)
        
    def forward(self, x: Tensor):
        x = self.ach(x)
        x = self.act(x)
        x = self.prj(x)
        return x
    
    @staticmethod
    def get_ex_channel(a: int) -> int:
        discriminant = 1 + 8*a
        
        x1 = (1 + math.sqrt(discriminant)) / 2
        x2 = (1 - math.sqrt(discriminant)) / 2
        
        return math.ceil(max(x1, x2))