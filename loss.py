"""
Rate-Distortion Loss for semantic idea compression.

    L = D + λ_nli · NLI + λ_rate · R_code + λ_gate · R_gate + λ_sem · CoSENT

Where:
    D        = cross-entropy distortion (token-level reconstruction)
    NLI      = differentiable NLI entailment penalty via facebook/bart-large-mnli.
               BART-large-mnli uses the same vocabulary as BART-base (V=50265).
               Gradient path: recon_logits → soft_probs → soft_hyp_embeds →
                              MNLI encoder → sentence_repr → -log p(entailment)
               λ_nli genuinely affects encoder/bottleneck gradients.
    R_code   = entropy model NLL for active slots (real information-theoretic bits)
    R_gate   = Bernoulli entropy of gate mask under a learned prior
    CoSENT   = optional sentence embedding contrastive loss (supplementary)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, BartForSequenceClassification

import config


class CoSENTLoss(nn.Module):
    def __init__(self, scale: float = 20.0):
        super().__init__()
        self.scale = scale

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        sim = torch.matmul(emb_a, emb_b.T)
        pos = torch.diag(sim)
        diff = (sim - pos.unsqueeze(1)) * self.scale
        mask = torch.eye(diff.size(0), device=diff.device).bool()
        diff = diff.masked_fill(mask, -1e4)
        return torch.log(1 + torch.exp(diff).sum()) / emb_a.size(0)


from typing import Union

class DifferentiableNLILoss(nn.Module):
    """
    Differentiable NLI entailment penalty using facebook/bart-large-mnli.

    Key insight: BART-large-mnli shares BART's BPE vocabulary (V=50265).
    This means soft token probabilities from BART-base's recon_logits can be
    projected through MNLI's embedding table into valid 1024-dim input embeddings
    — the same trick the CoSENT loss uses with SBERT, but now with genuine
    gradient flow through to the bottleneck.

    Architecture:
        Premise  (original text, discrete): prem_input_ids → embed_table → prem_emb
        Hypothesis (reconstruction, soft):  soft_probs     @ embed_table → hyp_emb
        Combined sequence: [prem_emb | EOS | hyp_emb | EOS]
        → BART-MNLI encoder → last-token repr → classification_head → p(entailment)

    We check only the forward direction during training:
        p(premise → hypothesis): did reconstruction preserve all of original's claims?
    The backward direction (hallucination check) is evaluated in eval_rd.py.

    Memory: BART-large-mnli ≈ 1.6GB. Frozen. Disable with LAMBDA_NLI = 0.0.
    """

    ENTAILMENT_IDX = 2  # BART-MNLI label order: 0=contradiction, 1=neutral, 2=entailment

    # MNLI was fine-tuned on sequences typically under 128 tokens total.
    # At 128 tokens per side we'd be at 258 combined — distribution shift territory.
    # Truncate each side to 60 tokens → 122 total, safely within MNLI's training range.
    MAX_TOKENS_PER_SIDE = 60

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        self.mnli = BartForSequenceClassification.from_pretrained(
            "facebook/bart-large-mnli"
        ).to(device)
        self.mnli.eval()
        for p in self.mnli.parameters():
            p.requires_grad = False

    def forward(
        self,
        recon_logits: torch.Tensor,        # [B, T, V] — BART-base reconstruction logits
        prem_input_ids: torch.Tensor,      # [B, T] — original token IDs (premise)
        prem_attention_mask: torch.Tensor, # [B, T]
        return_per_item: bool = False,     # if True, also return [B] p_entail per sample
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns scalar: mean -log p(entailment) over the batch.
        Gradient flows through recon_logits → MNLI encoder → bottleneck.

        If return_per_item=True, also returns p_entail [B] for scatter logging.
        """
        B, T_h, V = recon_logits.shape
        embed_table = self.mnli.model.shared.weight  # [V, 1024] — BART-large embedding

        # ── Truncate to stay within MNLI's training distribution ─────────────
        # Truncate premise: take the first MAX_TOKENS_PER_SIDE tokens.
        # Truncate hypothesis: take the first MAX_TOKENS_PER_SIDE tokens.
        # The beginning of each sequence carries the most semantic content
        # for a sentence-level NLI model (the main claim is typically upfront).
        K = self.MAX_TOKENS_PER_SIDE
        prem_emb_src = prem_input_ids[:, :K]          # [B, K]

        # ── Premise embeddings (frozen, no gradient) ──────────────────────────
        with torch.no_grad():
            prem_emb = embed_table[prem_emb_src]      # [B, K, 1024]

        # ── Soft hypothesis embeddings (differentiable) ───────────────────────
        # Truncate hypothesis logits to K tokens before the soft embedding projection.
        # soft_probs @ embed_table: [B, K, V] × [V, 1024] → [B, K, 1024]
        soft_probs = F.softmax(recon_logits[:, :K, :], dim=-1)
        hyp_emb = torch.matmul(soft_probs, embed_table)  # [B, K, 1024]

        # ── Construct combined sequence: [prem[:K] | EOS | hyp_soft[:K] | EOS] ─
        # Total length: K + 1 + K + 1 = 2K+2 = 122 tokens — within MNLI distribution.
        eos_id = self.mnli.config.eos_token_id
        eos_emb = embed_table[eos_id].view(1, 1, -1).expand(B, 1, -1)  # [B, 1, 1024]

        combined_emb = torch.cat([prem_emb, eos_emb, hyp_emb, eos_emb], dim=1)
        T_total = combined_emb.shape[1]  # 2K+2
        combined_mask = torch.ones(B, T_total, dtype=torch.long, device=self.device)

        # ── BART-large encoder (soft inputs, bypass EOS detection) ────────────
        encoder_out = self.mnli.model.encoder(
            inputs_embeds=combined_emb,
            attention_mask=combined_mask,
        )
        hidden = encoder_out.last_hidden_state        # [B, T_total, 1024]
        sentence_repr = hidden[:, -1, :]              # [B, 1024] — final EOS

        # ── Classification head → entailment probability ──────────────────────
        logits = self.mnli.classification_head(sentence_repr)          # [B, 3]
        p_entail = F.softmax(logits, dim=-1)[:, self.ENTAILMENT_IDX]  # [B]
        loss = -torch.log(p_entail.clamp(min=1e-8)).mean()

        if return_per_item:
            return loss, p_entail.detach()
        return loss


class RateDistortionLoss(nn.Module):
    """
    Full rate-distortion loss for semantic idea compression.

    Priority (by λ magnitude):
      1. NLI entailment  — ideas must survive (primary semantic constraint)
      2. CE distortion   — surface reconstruction fidelity
      3. Rate            — compress as aggressively as possible subject to the above
    """

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

        self.ce = nn.CrossEntropyLoss(ignore_index=-100)

        # Learned gate prior (Bernoulli gate entropy term)
        self.gate_prior_logit = nn.Parameter(torch.zeros(1))

        if config.LAMBDA_NLI > 0:
            self.nli_loss = DifferentiableNLILoss(device)
        else:
            self.nli_loss = None

        if config.LAMBDA_SEM > 0:
            self.sbert = AutoModel.from_pretrained(
                "sentence-transformers/all-distilroberta-v1"
            ).to(device)
            self.sbert.eval()
            for p in self.sbert.parameters():
                p.requires_grad = False
            self.cosent = CoSENTLoss()
        else:
            self.sbert = None

    def _mean_pool(self, model_out, attention_mask):
        tok_emb = model_out[0]
        mask = attention_mask.unsqueeze(-1).expand(tok_emb.size()).float()
        return torch.sum(tok_emb * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def _semantic_loss(self, recon_logits, input_ids, input_attention_mask):
        soft_probs = torch.softmax(recon_logits, dim=-1)
        sbert_emb_table = self.sbert.embeddings.word_embeddings.weight
        inputs_embeds = torch.matmul(soft_probs, sbert_emb_table)

        B, T, _ = inputs_embeds.shape
        dummy_mask = torch.ones(B, T, device=self.device)
        gen_out = self.sbert(inputs_embeds=inputs_embeds, attention_mask=dummy_mask)
        gen_emb = self._mean_pool(gen_out, dummy_mask)
        gen_emb = nn.functional.normalize(gen_emb, p=2, dim=1)

        with torch.no_grad():
            orig_out = self.sbert(input_ids=input_ids, attention_mask=input_attention_mask)
            orig_emb = self._mean_pool(orig_out, input_attention_mask)
            orig_emb = nn.functional.normalize(orig_emb, p=2, dim=1)

        return self.cosent(orig_emb, gen_emb)

    def forward(
        self,
        recon_logits: torch.Tensor,           # [B, T, vocab]
        gates: torch.Tensor,                  # [B, Q]
        labels: torch.Tensor,                 # [B, T]
        input_ids: torch.Tensor,              # [B, T]
        input_attention_mask: torch.Tensor,   # [B, T]
        flat_idx: torch.Tensor,               # [B, Q]
        log_probs: torch.Tensor,              # [B, Q]
        rate_scale: float = 0.0,
        return_nli_per_item: bool = False,    # for scatter logging: p_entail per sample
        gates_soft: torch.Tensor = None,      # [B, Q] smooth probabilities for gating loss
        nli_active: bool = True,              # disable in Stage 1 to save ~3x compute
    ):
        """
        Returns:
            total_loss, distortion, nli_loss, code_rate_loss, gate_rate_loss, sem_loss
            [, p_entail_per_item [B]] if return_nli_per_item=True
        """
        # ── Distortion ────────────────────────────────────────────────────────
        B, T, V = recon_logits.shape
        distortion = self.ce(recon_logits.reshape(-1, V), labels.reshape(-1))

        # ── NLI entailment (differentiable idea-completeness signal) ──────────
        p_entail_per_item = None
        if config.LAMBDA_NLI > 0 and self.nli_loss is not None and nli_active:
            if return_nli_per_item:
                nli, p_entail_per_item = self.nli_loss(
                    recon_logits, input_ids, input_attention_mask, return_per_item=True
                )
            else:
                nli = self.nli_loss(recon_logits, input_ids, input_attention_mask)
            nli_weighted = nli * config.LAMBDA_NLI
        else:
            nli = torch.tensor(0.0, device=self.device)
            nli_weighted = nli

        gate_target = gates_soft if gates_soft is not None else gates
        
        # ── Code rate (entropy model NLL, real bits) ──────────────────────────
        # Here we can safely use the hard 'gates' because it just drops inactive slots
        code_rate_nats = -(log_probs * gates).sum(dim=1).mean()
        code_rate_bits = code_rate_nats / math.log(2)
        code_rate_loss = code_rate_bits * config.LAMBDA_RATE * rate_scale

        # ── Gate rate (Bernoulli entropy under learned prior) ─────────────────
        p_gate = torch.sigmoid(self.gate_prior_logit).clamp(1e-6, 1 - 1e-6)
        gate_rate_bits = (
            -gate_target * torch.log2(p_gate)
            - (1 - gate_target) * torch.log2(1 - p_gate)
        ).sum(dim=1).mean()
        gate_rate_loss = gate_rate_bits * config.LAMBDA_GATE * rate_scale

        # ── Gate Warmup (prevent posterior collapse) ──────────────────────────
        # During Stage 1 (rate_scale == 0), the gate_net receives no gradient from
        # the rate penalties. Weight decay will push it to output 0, closing the gates.
        # This BCE loss explicitly forces the gate_net to output 1.0 (all gates open)
        # during Stage 1, so it is fully open and ready when Stage 2 begins.
        # It smoothly ramps down to 0 as rate_scale ramps up to 1.
        gate_warmup_loss = F.binary_cross_entropy(gate_target, torch.ones_like(gate_target)) * (1.0 - rate_scale)

        # ── Minimum-active-gates floor (hard anti-collapse) ───────────────────
        # Even with gate_warmup and ramp, mode collapse can win if the decoder
        # hasn't learned to use the bottleneck. This ReLU hinge penalises the
        # batch-mean active count dropping below MIN_GATES. Weight fades but
        # never vanishes while rate_scale < 2.0 (i.e. always on in Stages 1–2).
        MIN_GATES = 8
        avg_active = gate_target.sum(dim=1).mean()
        min_gate_loss = F.relu(MIN_GATES - avg_active) * 0.5 * (1.0 - rate_scale * 0.5)

        # ── Semantic CoSENT (supplementary) ───────────────────────────────────
        if config.LAMBDA_SEM > 0 and self.sbert is not None:
            sem_loss = self._semantic_loss(recon_logits, input_ids, input_attention_mask)
            sem_loss = sem_loss * config.LAMBDA_SEM
        else:
            sem_loss = torch.tensor(0.0, device=self.device)

        total = distortion + nli_weighted + code_rate_loss + gate_rate_loss + gate_warmup_loss + min_gate_loss + sem_loss
        if return_nli_per_item:
            return total, distortion, nli, code_rate_loss, gate_rate_loss, sem_loss, min_gate_loss, p_entail_per_item
        return total, distortion, nli, code_rate_loss, gate_rate_loss, sem_loss, min_gate_loss
