"""
Autoregressive Entropy Model over FSQ slot indices.

Models the joint distribution of Q FSQ codes as an autoregressive factorisation:
    p(z_1, ..., z_Q) = prod_i p(z_i | z_{<i})

During training: the log-likelihood under this model replaces the naive
  `active_gates × bits_per_code` rate term with a real information-theoretic rate.

During inference (with torchac): we arithmetic-code each slot index using the
  conditional CDF produced by this model, yielding actual compressed bytes.

Architecture: causal transformer encoder (GPT-style) with BOS prepended.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FSQEntropyModel(nn.Module):
    """
    Autoregressive transformer over Q FSQ slots.

    Args:
        num_queries:    Q — number of FSQ slots
        codebook_size:  product of FSQ levels (total number of flat indices)
        d_model:        transformer hidden dimension
        n_heads:        attention heads
        n_layers:       transformer layers
    """

    def __init__(
        self,
        num_queries: int,
        codebook_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.codebook_size = codebook_size
        self.bos_idx = codebook_size  # sentinel token prepended before the sequence

        # +1 embedding for BOS
        self.embed = nn.Embedding(codebook_size + 1, d_model)

        # Causal (GPT-style) transformer encoder.
        # We use TransformerEncoder with a causal mask — cleaner than Decoder for this use.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,    # pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Linear(d_model, codebook_size)

        # Pre-compute causal mask (Q × Q) so we don't recompute each forward
        self.register_buffer(
            "causal_mask",
            nn.Transformer.generate_square_subsequent_mask(num_queries),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    # ─── Core forward (teacher-forced) ───────────────────────────────────────

    def forward(self, flat_indices: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forced forward pass over all Q slots in parallel.

        Args:
            flat_indices: [B, Q] long — flat (mixed-radix) FSQ index per slot
        Returns:
            logits: [B, Q, codebook_size] — unnormalised log probs for each position
        """
        B = flat_indices.size(0)
        bos = torch.full((B, 1), self.bos_idx, dtype=torch.long, device=flat_indices.device)
        # Shift right: BOS + indices[:-1]
        inp = torch.cat([bos, flat_indices[:, :-1]], dim=1)  # [B, Q]
        x = self.embed(inp)                                    # [B, Q, d_model]
        out = self.transformer(x, mask=self.causal_mask, is_causal=True)  # [B, Q, d_model]
        return self.head(out)                                  # [B, Q, codebook_size]

    # ─── Training helpers ─────────────────────────────────────────────────────

    def log_prob(self, flat_indices: torch.Tensor) -> torch.Tensor:
        """
        Per-slot log-probability (nats) of observed indices under the autoregressive prior.

        Args:
            flat_indices: [B, Q] long
        Returns:
            log_probs: [B, Q] float  (in nats)
        """
        logits = self.forward(flat_indices)                        # [B, Q, C]
        log_probs = F.log_softmax(logits, dim=-1)                  # [B, Q, C]
        return log_probs.gather(-1, flat_indices.unsqueeze(-1)).squeeze(-1)  # [B, Q]

    # ─── Inference helpers (for arithmetic coding) ────────────────────────────

    def step_cdf(self, decoded_so_far: torch.Tensor) -> torch.Tensor:
        """
        Compute the CDF for the *next* slot index given previously decoded slots.
        Used for sequential arithmetic decode.

        Args:
            decoded_so_far: [B, n] long — n indices decoded so far (n may be 0)
        Returns:
            cdf: [B, codebook_size + 1] float in [0, 1]
        """
        B = decoded_so_far.size(0)
        n = decoded_so_far.size(1)
        device = decoded_so_far.device

        bos = torch.full((B, 1), self.bos_idx, dtype=torch.long, device=device)
        inp = torch.cat([bos, decoded_so_far], dim=1)   # [B, n+1]
        x = self.embed(inp)                              # [B, n+1, d_model]

        # Causal mask for the current prefix length
        mask = self.causal_mask[: n + 1, : n + 1]
        out = self.transformer(x, mask=mask, is_causal=True)  # [B, n+1, d_model]
        logits = self.head(out[:, -1, :])                      # [B, C] — predict next
        probs = F.softmax(logits, dim=-1)                      # [B, C]

        # Prepend 0 and cumsum to form CDF [B, C+1]
        cdf = torch.cat(
            [torch.zeros(B, 1, device=device), probs.cumsum(dim=-1)],
            dim=-1,
        ).clamp(0.0, 1.0)
        return cdf

    def teacher_forced_cdfs(self, flat_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute CDFs for all Q slots in one parallel forward pass (for encoding).

        Args:
            flat_indices: [B, Q] long
        Returns:
            cdfs: [B, Q, codebook_size + 1] float in [0, 1]
        """
        logits = self.forward(flat_indices)           # [B, Q, C]
        probs = F.softmax(logits, dim=-1)             # [B, Q, C]
        B, Q, C = probs.shape
        cdfs = torch.cat(
            [torch.zeros(B, Q, 1, device=probs.device), probs.cumsum(dim=-1)],
            dim=-1,
        ).clamp(0.0, 1.0)
        return cdfs                                    # [B, Q, C+1]
