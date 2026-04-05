"""
Finite Scalar Quantization (FSQ)
Mentzer et al., NeurIPS 2023 — https://arxiv.org/abs/2309.15505

Drop-in replacement for VQ-VAE and Gumbel-Softmax bottlenecks.
Key advantages over Gumbel-Softmax:
  - No codebook collapse (no commitment loss needed)
  - Stable gradients (straight-through over bounded tanh, not 50k softmax classes)
  - Precisely controlled bit budget: bits = sum(log2(level) for level in levels)
  - No temperature annealing required

Recommended level configs:
  [8, 5, 5, 5]     -> ~10.0 bits/code  (good default for text)
  [8, 8, 8, 8]     -> ~12.0 bits/code  (higher fidelity)
  [8, 8, 8, 6, 5]  -> ~13.9 bits/code  (near-lossless)
  [8, 8, 8]        ->  ~9.0 bits/code  (aggressive compression)
"""

import math
import torch
import torch.nn as nn


class FiniteScalarQuantizer(nn.Module):
    def __init__(self, levels: list[int]):
        """
        Args:
            levels: list of integers, one per FSQ dimension.
                    e.g. [8, 5, 5, 5] quantizes 4 dims to 8, 5, 5, 5 levels each.
        """
        super().__init__()
        self.levels = levels
        self.num_dims = len(levels)

        levels_t = torch.tensor(levels, dtype=torch.float32)
        self.register_buffer("levels_t", levels_t)

        half_l = (levels_t - 1) / 2.0
        self.register_buffer("half_l", half_l)

        self.bits_per_code = sum(math.log2(l) for l in levels)
        self.codebook_size = int(torch.prod(levels_t).item())

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        """Bound z to (-1, 1) via tanh."""
        return torch.tanh(z)

    def quantize(self, z_bounded: torch.Tensor) -> torch.Tensor:
        """Round bounded values to nearest quantization level (no gradient)."""
        return (torch.round(z_bounded * self.half_l) / self.half_l).clamp(-1.0, 1.0)

    def forward(self, z: torch.Tensor):
        """
        Quantize with straight-through gradient.

        Args:
            z: [..., num_dims] continuous encoder output

        Returns:
            z_q:  [..., num_dims] quantized, gradient flows through (for training)
            z_qh: [..., num_dims] hard quantized, no gradient (for rate computation)
        """
        z_bounded = self.bound(z)
        z_qh = self.quantize(z_bounded)
        z_q = z_bounded + (z_qh - z_bounded).detach()
        return z_q, z_qh

    def to_indices(self, z: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous codes to per-dimension discrete indices.

        Returns:
            indices: [..., num_dims] long tensor, each dim in [0, level-1]
        """
        z_bounded = self.bound(z)
        indices = torch.round(z_bounded * self.half_l).long() + self.half_l.long()
        # clamp() with int min and Tensor max fails in newer PyTorch versions
        indices = indices.clamp_min(0)
        return torch.min(indices, (self.levels_t - 1).long())

    def from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct quantized continuous codes from discrete indices.

        Args:
            indices: [..., num_dims] long tensor, each dim in [0, level-1]

        Returns:
            z_q: [..., num_dims] in (-1, 1)
        """
        return ((indices.float() - self.half_l) / self.half_l).clamp(-1.0, 1.0)

    def to_flat_index(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Convert per-dimension indices to a single flat integer (mixed-radix).
        Useful for serialisation.
        """
        multipliers = torch.ones(self.num_dims, dtype=torch.long, device=indices.device)
        for i in range(self.num_dims - 2, -1, -1):
            multipliers[i] = multipliers[i + 1] * self.levels[i + 1]
        return (indices * multipliers).sum(dim=-1)

    def from_flat_index(self, flat: torch.Tensor) -> torch.Tensor:
        """Inverse of to_flat_index."""
        indices = torch.zeros(
            (*flat.shape, self.num_dims), dtype=torch.long, device=flat.device
        )
        remainder = flat.clone()
        for i in range(self.num_dims - 1, -1, -1):
            indices[..., i] = remainder % self.levels[i]
            remainder = remainder // self.levels[i]
        return indices

    def extra_repr(self) -> str:
        return (
            f"levels={self.levels}, "
            f"codebook_size={self.codebook_size}, "
            f"bits_per_code={self.bits_per_code:.2f}"
        )
