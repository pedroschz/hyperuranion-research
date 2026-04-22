"""
SemanticAutoencoder — FSQ + Autoregressive Entropy Model revision

Architecture:
  1. BART encoder + LoRA adapters  ->  contextual hidden states
  2. Iterative Perceiver (K rounds) ->  fixed pool of NUM_QUERIES latent slots
  3. Linear projection + FSQ        ->  discrete codes (~10 bits/code, stable)
  4. Learned gate network           ->  variable-length: zeros out unneeded slots
  5. FSQEntropyModel                ->  AR prior over slot indices (real rate)
  6. Linear unproject + BART dec.   ->  reconstructed text logits

Key changes from v1:
  - Perceiver cross-attention is now iterated K times (Perceiver IO)
  - FSQEntropyModel provides the rate signal (entropy NLL) instead of slot count
  - compress() / decompress() use the ArithmeticCodec for real byte streams
"""

import math
import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right
from transformers.modeling_outputs import BaseModelOutput
from peft import LoraConfig, get_peft_model, TaskType

import config
from fsq import FiniteScalarQuantizer
from entropy_model import FSQEntropyModel
import codec


class SemanticAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        # ── 1. BART backbone + LoRA ────────────────────────────────────────────
        bart = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        lora_cfg = LoraConfig(
            r=config.LORA_RANK,
            lora_alpha=config.LORA_ALPHA,
            target_modules=["q_proj", "v_proj", "out_proj", "fc1", "fc2"],
            lora_dropout=config.LORA_DROPOUT,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        self.bart = get_peft_model(bart, lora_cfg)
        self.bart.print_trainable_parameters()

        # BartConfig doesn't expose mask_token_id; it lives on the tokenizer.
        # For facebook/bart-base this is 50264. Cache once here so word-dropout
        # doesn't need a tokenizer round-trip per step.
        from transformers import BartTokenizer
        self.mask_token_id = BartTokenizer.from_pretrained("facebook/bart-base").mask_token_id

        H = self.bart.get_base_model().config.hidden_size  # 768 for bart-base

        # ── 2. Iterative Perceiver cross-attention ─────────────────────────────
        self.num_queries = config.NUM_QUERIES
        self.perceiver_iters = config.PERCEIVER_ITERS
        self.query_tokens = nn.Parameter(torch.randn(1, self.num_queries, H) * 0.02)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=H, num_heads=8, batch_first=True, dropout=0.1
        )
        # LayerNorm + FFN for post-attention residual updates inside the perceiver loop
        self.perceiver_ln1 = nn.LayerNorm(H)
        self.perceiver_ln2 = nn.LayerNorm(H)
        self.perceiver_ffn = nn.Sequential(
            nn.Linear(H, H * 4),
            nn.GELU(),
            nn.Linear(H * 4, H),
        )

        # ── 3. FSQ bottleneck ──────────────────────────────────────────────────
        # LayerNorm stabilises pooled-slot magnitudes before the 768→4 squeeze,
        # so fsq_proj's gradient signal isn't dominated by slot-norm drift.
        self.fsq_pre_ln = nn.LayerNorm(H)
        self.fsq_proj = nn.Linear(H, config.FSQ_DIMS)
        # std=0.05: with LayerNorm input (unit std, 768 dims), output per dim ≈
        # 0.05 * sqrt(768) ≈ 1.38 → tanh(1.38) ≈ 0.88 — codes use ~88% of FSQ
        # range without saturating. std=0.1 was too large (z_in→60, full saturation).
        nn.init.normal_(self.fsq_proj.weight, std=0.05)
        nn.init.zeros_(self.fsq_proj.bias)
        self.fsq = FiniteScalarQuantizer(levels=config.FSQ_LEVELS)
        self.fsq_unproj = nn.Linear(config.FSQ_DIMS, H)

        # ── 4. Learned gate network ────────────────────────────────────────────
        self.gate_net = nn.Sequential(
            nn.Linear(H, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )
        # Start gates half-open (sigmoid(0) = 0.5) with a small-magnitude weight,
        # so early gate decisions depend on learned signal, not init noise.
        nn.init.zeros_(self.gate_net[-1].bias)
        nn.init.normal_(self.gate_net[-1].weight, std=0.01)

        # ── 5. Autoregressive entropy model ────────────────────────────────────
        self.entropy_model = FSQEntropyModel(
            num_queries=config.NUM_QUERIES,
            codebook_size=self.fsq.codebook_size,
            d_model=config.ENTROPY_MODEL_DIM,
            n_heads=config.ENTROPY_MODEL_HEADS,
            n_layers=config.ENTROPY_MODEL_LAYERS,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _encode_to_slots(self, input_ids, attention_mask=None):
        """
        Run encoder + iterative Perceiver, return pooled hidden states (pre-FSQ).
        Returns:
            pooled: [B, Q, H]
        """
        base = self.bart.get_base_model()
        enc_out = base.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = enc_out.last_hidden_state  # [B, T, H]

        B = hidden.size(0)
        queries = self.query_tokens.expand(B, -1, -1).contiguous()
        key_pad_mask = (attention_mask == 0) if attention_mask is not None else None

        # Iterative refinement: queries attend to encoder hidden states K times.
        for _ in range(self.perceiver_iters):
            delta, _ = self.cross_attention(
                query=queries,
                key=hidden.contiguous(),
                value=hidden.contiguous(),
                key_padding_mask=key_pad_mask,
            )  # [B, Q, H]
            queries = self.perceiver_ln1(queries + delta)
            queries = self.perceiver_ln2(queries + self.perceiver_ffn(queries))

        return queries  # [B, Q, H]

    def _quantise(self, pooled):
        """
        Project pooled slots through FSQ.
        Returns:
            z_q:        [B, Q, FSQ_DIMS]  quantised (straight-through gradient)
            z_qh:       [B, Q, FSQ_DIMS]  hard quantised (for rate / codec)
            gates:      [B, Q]            hard gates (0 or 1) with straight-through gradient
            gates_soft: [B, Q]            soft gate probabilities (for loss)
            flat_idx:   [B, Q]            long — flat mixed-radix index per slot
        """
        z_in = self.fsq_proj(self.fsq_pre_ln(pooled)).clamp(-3.0, 3.0)  # [B, Q, FSQ_DIMS]
        z_q, z_qh = self.fsq(z_in)            # both [B, Q, FSQ_DIMS]

        # Derive per-dim integer indices from z_in (same values as z_qh, but integer).
        # Detach so the flat_idx used by the entropy model log-prob doesn't create
        # a double-backward through the quantization step.
        with torch.no_grad():
            per_dim_idx = self.fsq.to_indices(z_in)      # [B, Q, FSQ_DIMS] long
            flat_idx = self.fsq.to_flat_index(per_dim_idx)  # [B, Q] long

        gate_logits = self.gate_net(pooled).squeeze(-1)  # [B, Q]
        gates_soft = torch.sigmoid(gate_logits)          # [B, Q]
        
        # Straight-Through Estimator (STE) for gates:
        # Forward pass sees exact 0s and 1s, but gradients flow through gates_soft.
        # This prevents the "magnitude shift" collapse when gates drop from 1.0 to 0.5.
        gates_hard = (gates_soft > config.GATE_THRESHOLD).float()
        gates = gates_hard.detach() - gates_soft.detach() + gates_soft
        
        return z_q, z_qh, gates, gates_soft, flat_idx

    def _decode(self, z_q, gates, decoder_input_ids):
        """
        Decode quantised codes to logits via BART decoder.
        Args:
            z_q:               [B, Q, FSQ_DIMS]
            gates:             [B, Q]  (hard during train/inference)
            decoder_input_ids: [B, T]
        Returns:
            logits: [B, T, vocab_size]
        """
        base = self.bart.get_base_model()
        enc_hidden = self.fsq_unproj(z_q)   # [B, Q, H]

        # Attention mask exactly matches the hard gates.
        # 1 means attend, 0 means ignore.
        enc_mask = gates.long()

        dec_out = base.get_decoder()(
            input_ids=decoder_input_ids,
            encoder_hidden_states=enc_hidden,
            encoder_attention_mask=enc_mask,
            return_dict=True,
        )
        return base.lm_head(dec_out.last_hidden_state)  # [B, T, vocab]

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, input_ids, attention_mask=None, labels=None, rate_scale=1.0):
        """
        Full autoencoder forward pass (training).

        Returns:
            recon_logits: [B, T, vocab]
            gates:        [B, Q]  hard gate probabilities (0 or 1) with STE gradient
            gates_soft:   [B, Q]  soft probabilities (for loss computation)
            z_qh:         [B, Q, FSQ_DIMS]  hard codes
            flat_idx:     [B, Q]  long  — flat FSQ index per slot
            log_probs:    [B, Q]  float — per-slot log-prob under entropy model (nats)
        """
        base = self.bart.get_base_model()

        pooled = self._encode_to_slots(input_ids, attention_mask)
        z_q, z_qh, gates, gates_soft, flat_idx = self._quantise(pooled)

        # Prevent posterior collapse in Stage 1: if rate_scale is 0,
        # force the decoder to attend to all slots.
        # This forces the encoder/bottleneck to learn useful representations
        # rather than letting the gate network take the easy way out and shut off.
        decoder_gates = torch.ones_like(gates) if rate_scale == 0.0 else gates

        # Entropy model: compute rate signal (teacher-forced, in nats)
        log_probs = self.entropy_model.log_prob(flat_idx)  # [B, Q]

        target = labels if labels is not None else input_ids
        dec_ids = shift_tokens_right(
            target,
            base.config.pad_token_id,
            base.config.decoder_start_token_id,
        )

        # Prevent posterior collapse (Bowman et al. 2015 Word Dropout):
        # Randomly replace decoder input tokens with <mask> during training.
        # This breaks the autoregressive LM shortcut — the decoder CANNOT
        # reconstruct masked tokens from local context alone, so it is
        # forced to attend to the quantised bottleneck for information.
        if self.training and getattr(config, "WORD_DROPOUT", 0.0) > 0.0:
            dropout_mask = torch.rand(dec_ids.shape, device=dec_ids.device) < config.WORD_DROPOUT
            dropout_mask[:, 0] = False  # never mask the BOS / decoder start token
            dropout_mask &= (dec_ids != base.config.pad_token_id)  # never mask padding
            dec_ids = dec_ids.masked_fill(dropout_mask, self.mask_token_id)

        recon_logits = self._decode(z_q, decoder_gates, dec_ids)
        return recon_logits, gates, gates_soft, z_qh, flat_idx, log_probs

    def compress(self, input_ids, attention_mask=None):
        """
        Compress text to a byte string using the learned entropy model.

        Returns:
            payload:   bytes  — the actual compressed bitstream
            gate_mask: [B, Q] bool  — for inspection / statistics
            num_bits:  float  — actual bit count of payload
            entropy_bits: float — entropy model's estimate (theoretical lower bound)
        """
        with torch.no_grad():
            pooled = self._encode_to_slots(input_ids, attention_mask)
            z_in = self.fsq_proj(self.fsq_pre_ln(pooled)).clamp(-3.0, 3.0)
            indices = self.fsq.to_indices(z_in)          # [B, Q, FSQ_DIMS]
            flat_idx = self.fsq.to_flat_index(indices)   # [B, Q]

            gate_logits = self.gate_net(pooled).squeeze(-1)
            gates_soft = torch.sigmoid(gate_logits)
            gate_mask = gates_soft > config.GATE_THRESHOLD    # [B, Q] bool

            # Theoretical rate from entropy model (for logging)
            log_probs = self.entropy_model.log_prob(flat_idx)          # [B, Q]
            active = gate_mask.float()
            entropy_nats = -(log_probs * active).sum(dim=1).mean()
            entropy_bits = (entropy_nats / math.log(2)).item()

        # Produce real bytes (B=1 assumed at inference)
        payload = codec.compress(flat_idx[:1], gate_mask[:1], self.entropy_model)
        num_bits = codec.payload_bits(payload)

        return payload, gate_mask, num_bits, entropy_bits

    def decompress(self, payload, max_length=128, num_beams=4, **kwargs):
        """
        Reconstruct text from a compressed byte string.

        Args:
            payload: bytes produced by compress()
        Returns:
            generated_ids: [1, T]
        """
        device = next(self.parameters()).device
        base = self.bart.get_base_model()

        flat_idx, gate_mask = codec.decompress(
            payload,
            Q=self.num_queries,
            entropy_model=self.entropy_model,
            device=device,
        )

        # Recover per-dim indices, then continuous codes
        indices = self.fsq.from_flat_index(flat_idx)      # [1, Q, FSQ_DIMS]
        z_q = self.fsq.from_indices(indices)              # [1, Q, FSQ_DIMS]
        enc_hidden = self.fsq_unproj(z_q)                 # [1, Q, H]
        
        # Apply gate_mask exactly like training
        enc_mask = gate_mask.long()

        enc_outputs = BaseModelOutput(last_hidden_state=enc_hidden)
        return base.generate(
            encoder_outputs=enc_outputs,
            attention_mask=enc_mask,
            max_length=max_length,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            **kwargs,
        )

    # Convenience: compress/decompress keeping indices visible (for eval_rd.py)
    def compress_to_indices(self, input_ids, attention_mask=None):
        """
        Like compress() but returns indices + gate_mask for inspection,
        in addition to the byte payload.
        """
        with torch.no_grad():
            pooled = self._encode_to_slots(input_ids, attention_mask)
            z_in = self.fsq_proj(self.fsq_pre_ln(pooled)).clamp(-3.0, 3.0)
            indices = self.fsq.to_indices(z_in)
            flat_idx = self.fsq.to_flat_index(indices)
            gate_logits = self.gate_net(pooled).squeeze(-1)
            gates_soft = torch.sigmoid(gate_logits)
            gate_mask = gates_soft > config.GATE_THRESHOLD
            log_probs = self.entropy_model.log_prob(flat_idx)
            active = gate_mask.float()
            entropy_nats = -(log_probs * active).sum(dim=1).mean()
            entropy_bits = (entropy_nats / math.log(2)).item()

        payload = codec.compress(flat_idx[:1], gate_mask[:1], self.entropy_model)
        num_bits = codec.payload_bits(payload)
        return indices, gate_mask, payload, num_bits, entropy_bits

    def decompress_from_indices(self, indices, gate_mask, max_length=128, num_beams=4, **kwargs):
        """Decompress from raw indices + gate_mask (no byte parsing). Used by eval_rd.py."""
        base = self.bart.get_base_model()
        z_q = self.fsq.from_indices(indices)       # [B, Q, FSQ_DIMS]
        enc_hidden = self.fsq_unproj(z_q)          # [B, Q, H]
        
        # Apply gate_mask exactly like training
        enc_mask = gate_mask.long()
        
        enc_outputs = BaseModelOutput(last_hidden_state=enc_hidden)

        return base.generate(
            encoder_outputs=enc_outputs,
            attention_mask=enc_mask,
            max_length=max_length,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            **kwargs,
        )
