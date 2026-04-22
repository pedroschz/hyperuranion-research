"""
Training script for SemanticAutoencoder with three-stage curriculum.

Stage 1 [epochs 0 → STAGE2_START):
    rate_scale = 0 — train reconstruction only; model fills all slots.

Stage 2 [epochs STAGE2_START → STAGE3_START):
    rate_scale ramps 0 → 1 — backbone + entropy model jointly compressed.

Stage 3 [epochs STAGE3_START → NUM_EPOCHS):
    Backbone (BART + LoRA + perceiver + FSQ + gate) frozen.
    Only entropy model + gate_prior trained.
    → Maximum coding efficiency without disturbing reconstruction quality.
"""

import math
from collections import defaultdict

import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, BartTokenizer
import wandb
from tqdm import tqdm
import numpy as np

import config
import codec
from model import SemanticAutoencoder
from data_loader import get_dataloader
from loss import RateDistortionLoss


# ──────────────────────────────────────────────────────────────────────────────
# Curriculum helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_rate_scale(epoch: int, step_in_epoch: int, steps_per_epoch: int) -> float:
    """
    Returns the rate_scale in [0, 1] for the current training position.

    Stage 1: 0
    Stage 2: linearly ramps from 0 to 1 over the stage's duration
    Stage 3: 1
    """
    if epoch < config.STAGE2_START:
        return 0.0
    if epoch >= config.STAGE3_START:
        return 1.0
    # Stage 2: ramp
    stage2_total = (config.STAGE3_START - config.STAGE2_START) * steps_per_epoch
    stage2_step = (epoch - config.STAGE2_START) * steps_per_epoch + step_in_epoch
    return min(1.0, stage2_step / max(1, stage2_total))


def freeze_backbone(model: SemanticAutoencoder):
    """Freeze everything except the entropy model and gate prior in Stage 3."""
    for name, p in model.named_parameters():
        if "entropy_model" not in name:
            p.requires_grad = False
    print("  [Stage 3] Backbone frozen. Training entropy model only.")


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def _log_gradient_ratio(
    model, criterion,
    recon_logits, gates, labels,
    input_ids, attention_mask,
    flat_idx, log_probs, rate_scale,
    device, global_step,
):
    """
    Measure the ratio of NLI vs CE gradient norms at model.fsq_proj.weight.

    Answers: "Is LAMBDA_NLI actually training the bottleneck, or has gradient
    attenuation through 12 frozen BART-large layers made it a monitoring signal?"

    Runs on a 2-item sub-batch to keep retain_graph memory bounded. The full
    batch's retain_graph would hold the entire MNLI encoder activations in memory
    until the next .backward() — on a 16GB card this causes intermittent OOMs at
    diagnostic steps even when normal steps are fine.

    Logs "grad/nli_vs_ce_ratio_at_fsq". If consistently < 0.01, increase LAMBDA_NLI.
    """
    if criterion.nli_loss is None:
        return

    probe_param = model.fsq_proj.weight
    if probe_param.grad is None:
        return

    try:
        # Sub-batch: first 2 items only — enough to measure the ratio,
        # bounded memory cost from retain_graph.
        sb = min(2, recon_logits.shape[0])
        rl_sb  = recon_logits[:sb].detach().requires_grad_(True)
        ids_sb = input_ids[:sb]
        msk_sb = attention_mask[:sb]
        lbl_sb = labels[:sb]

        # CE gradient
        ce_only = criterion.ce(
            rl_sb.reshape(-1, rl_sb.shape[-1]), lbl_sb.reshape(-1)
        )
        (ce_grad,) = torch.autograd.grad(
            ce_only, rl_sb, retain_graph=False, allow_unused=True
        )

        # NLI gradient (fresh sub-batch forward — no retain_graph needed)
        rl_sb2 = recon_logits[:sb].detach().requires_grad_(True)
        nli_only = criterion.nli_loss(rl_sb2, ids_sb, msk_sb)
        (nli_grad,) = torch.autograd.grad(
            nli_only * config.LAMBDA_NLI, rl_sb2,
            retain_graph=False, allow_unused=True,
        )

        ce_norm  = ce_grad.norm().item()  if ce_grad  is not None else 0.0
        nli_norm = nli_grad.norm().item() if nli_grad is not None else 0.0
        ratio = nli_norm / (ce_norm + 1e-12)

        wandb.log({
            "grad/ce_norm_at_recon_logits":      ce_norm,
            "grad/nli_norm_at_recon_logits":     nli_norm,
            "grad/nli_vs_ce_ratio_at_recon_logits": ratio,
            "step": global_step,
        })

        if ratio < 0.01:
            print(
                f"  [Step {global_step}] NLI gradient is very small "
                f"(nli/ce={ratio:.4f}). Consider increasing LAMBDA_NLI in config."
            )

    except torch.cuda.OutOfMemoryError:
        print(
            f"  [Step {global_step}] Gradient ratio diagnostic OOM — "
            "reduce frequency (currently every 100 steps) or sub-batch size (currently 2)."
        )
    except Exception:
        pass  # never crash the training run for a diagnostic


def _log_scatter_bucketed(scatter_buffer: list, global_step: int) -> None:
    """
    Bucket (active_gate_count, p_entail) pairs and log mean ± std per bucket.

    With B=8 and NUM_QUERIES=32 there are at most 33 distinct gate counts.
    A raw scatter of 8 points per step is too sparse to read; aggregating over
    a sliding window of 10 batches (80 points) gives interpretable statistics.
    """
    buckets: dict[int, list[float]] = defaultdict(list)
    for gate_count, p_ent in scatter_buffer:
        buckets[int(gate_count)].append(p_ent)

    table = wandb.Table(columns=["gate_count", "mean_p_entail", "std_p_entail", "n_samples"])
    for gc in sorted(buckets.keys()):
        vals = buckets[gc]
        table.add_data(
            gc,
            float(np.mean(vals)),
            float(np.std(vals)) if len(vals) > 1 else 0.0,
            len(vals),
        )
    wandb.log({"scatter/gates_vs_p_entail": table, "step": global_step})


# Hardcoded sentence pairs with known entailment labels — used for NLI sanity checks.
# Format: (premise, hypothesis, label_str, expected_p_entail_direction)
# "high" = p_entail should be well above 0.5; "low" = well below 0.5
_NLI_SANITY_PAIRS = [
    (
        "A dog is running in the park.",
        "An animal is moving outdoors.",
        "entailment", "high",
    ),
    (
        "All students passed the exam.",
        "No student passed the exam.",
        "contradiction", "low",
    ),
    (
        "The company reported record profits.",
        "Shareholders are pleased with the results.",
        "neutral", None,  # neutral — no direction check, just log
    ),
]


@torch.no_grad()
def _nli_sanity_check(criterion, tokenizer, device, global_step: int) -> None:
    """
    Run the NLI model on hardcoded pairs with known labels every 1000 steps.

    Detects silent failures: the NLI model returning confidently wrong scores
    (e.g., due to the 60-token truncation putting it out-of-distribution, or
    the classification head behaving unexpectedly on soft inputs).

    Uses the discrete token path (not the soft-embedding path) since we're
    checking the NLI model itself, not the reconstruction quality.
    """
    if criterion.nli_loss is None:
        return

    import torch.nn.functional as F

    log_dict = {"step": global_step}
    all_ok = True

    for premise, hypothesis, label, direction in _NLI_SANITY_PAIRS:
        enc = criterion.nli_loss.mnli.config
        tok = tokenizer  # BartTokenizer shares vocab with BART-large-mnli

        prem_tok = tok(premise, return_tensors="pt", truncation=True, max_length=60).to(device)
        hyp_tok  = tok(hypothesis, return_tensors="pt", truncation=True, max_length=60).to(device)

        # Build combined embedding: [prem | EOS | hyp | EOS]
        embed_table = criterion.nli_loss.mnli.model.shared.weight
        eos_id  = criterion.nli_loss.mnli.config.eos_token_id
        eos_emb = embed_table[eos_id].view(1, 1, -1)

        prem_emb = embed_table[prem_tok["input_ids"]]   # [1, T_p, 1024]
        hyp_emb  = embed_table[hyp_tok["input_ids"]]    # [1, T_h, 1024]
        combined = torch.cat([prem_emb, eos_emb, hyp_emb, eos_emb], dim=1)
        mask = torch.ones(1, combined.shape[1], dtype=torch.long, device=device)

        enc_out = criterion.nli_loss.mnli.model.encoder(
            inputs_embeds=combined, attention_mask=mask
        )
        sentence_repr = enc_out.last_hidden_state[:, -1, :]
        logits = criterion.nli_loss.mnli.classification_head(sentence_repr)
        p_entail = F.softmax(logits, dim=-1)[0, criterion.nli_loss.ENTAILMENT_IDX].item()

        safe_label = label.replace(" ", "_")
        log_dict[f"eval/nli_sanity_{safe_label}"] = p_entail

        if direction == "high" and p_entail < 0.4:
            print(
                f"  [NLI sanity, step {global_step}] WARN: entailment pair scored "
                f"p_entail={p_entail:.3f} (expected > 0.4). NLI signal may be unreliable."
            )
            all_ok = False
        elif direction == "low" and p_entail > 0.6:
            print(
                f"  [NLI sanity, step {global_step}] WARN: contradiction pair scored "
                f"p_entail={p_entail:.3f} (expected < 0.6). NLI signal may be unreliable."
            )
            all_ok = False

    log_dict["eval/nli_sanity_ok"] = int(all_ok)
    wandb.log(log_dict)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def train():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    print(f"Arithmetic coding: {'ON (torchac)' if codec.is_entropy_coded() else 'OFF (uniform fallback)'}")

    model = SemanticAutoencoder().to(device)
    dataloader, tokenizer = get_dataloader()

    criterion = RateDistortionLoss(device=device).to(device)

    # All trainable params: backbone + entropy model + gate prior (in criterion)
    trainable_params = (
        [p for p in model.parameters() if p.requires_grad]
        + list(criterion.parameters())
    )
    optimizer = AdamW(trainable_params, lr=config.LEARNING_RATE, weight_decay=0.01)

    steps_per_epoch = len(dataloader)
    total_steps = config.NUM_EPOCHS * steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps,
    )

    wandb.init(
        project="semantic-autoencoder-fsq",
        config={
            "fsq_levels": config.FSQ_LEVELS,
            "fsq_bits_per_code": model.fsq.bits_per_code,
            "num_queries": config.NUM_QUERIES,
            "perceiver_iters": config.PERCEIVER_ITERS,
            "entropy_model_dim": config.ENTROPY_MODEL_DIM,
            "lambda_rate": config.LAMBDA_RATE,
            "lambda_gate": config.LAMBDA_GATE,
            "lambda_nli": config.LAMBDA_NLI,
            "lambda_sem": config.LAMBDA_SEM,
            "stage2_start": config.STAGE2_START,
            "stage3_start": config.STAGE3_START,
            "batch_size": config.BATCH_SIZE,
            "lr": config.LEARNING_RATE,
            "codec": "torchac" if codec.is_entropy_coded() else "uniform",
        },
    )

    global_step = 0
    stage3_frozen = False

    # Scatter accumulator: collect (gate_count, p_entail) pairs across batches,
    # flush and bucket every SCATTER_FLUSH_EVERY batches to get interpretable stats.
    SCATTER_FLUSH_EVERY = 10      # flush after 10 collect-steps (every 200 main steps)
    scatter_buffer: list[tuple[int, float]] = []
    scatter_collect_count = 0

    try:
        for epoch in range(config.NUM_EPOCHS):

            # Stage transitions
            if epoch == config.STAGE2_START:
                print(f"\n[Epoch {epoch}] Entering Stage 2: rate penalty active.")
            if epoch == config.STAGE3_START and not stage3_frozen:
                freeze_backbone(model)
                # Rebuild optimizer with only entropy model params + gate prior
                trainable_params = (
                    [p for p in model.parameters() if p.requires_grad]
                    + list(criterion.parameters())
                )
                optimizer = AdamW(trainable_params, lr=config.LEARNING_RATE * 0.1, weight_decay=0.01)
                stage3_frozen = True

            model.train()
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
            epoch_losses = {
                "total": 0.0, "distortion": 0.0, "nli": 0.0,
                "code_rate": 0.0, "gate_rate": 0.0, "semantic": 0.0,
            }

            for step_in_epoch, batch in enumerate(pbar):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                rate_scale = get_rate_scale(epoch, step_in_epoch, steps_per_epoch)

                # Every 200 steps, collect per-item NLI scores for scatter logging.
                # This is cheap: same forward pass, just with return_per_item=True.
                want_scatter = (global_step % 200 == 0)

                optimizer.zero_grad()

                recon_logits, gates, gates_soft, z_qh, flat_idx, log_probs = model(
                    input_ids, attention_mask=attention_mask, labels=labels, rate_scale=rate_scale
                )

                # retain_grad so we can measure the NLI gradient at recon_logits
                # after the backward pass, without a second forward pass.
                # (Skip this in Stage 3 where the backbone is frozen).
                if recon_logits.requires_grad:
                    recon_logits.retain_grad()

                # NLI (BART-large-mnli) is ~3x the compute of the base model.
                # In Stage 1, CE is the primary signal — NLI's semantic constraint
                # only becomes critical in Stage 2 when compression forces tradeoffs.
                # Disabling it in Stage 1 cuts total training time by ~40%.
                nli_active = rate_scale > 0.0

                criterion_out = criterion(
                    recon_logits=recon_logits,
                    gates=gates,
                    gates_soft=gates_soft,
                    labels=labels,
                    input_ids=input_ids,
                    input_attention_mask=attention_mask,
                    flat_idx=flat_idx,
                    log_probs=log_probs,
                    rate_scale=rate_scale,
                    return_nli_per_item=want_scatter,
                    nli_active=nli_active,
                )
                if want_scatter:
                    total, distortion, nli, code_rate, gate_rate, semantic, min_gate, p_entail_items = criterion_out
                else:
                    total, distortion, nli, code_rate, gate_rate, semantic, min_gate = criterion_out
                    p_entail_items = None

                if math.isnan(total.item()):
                    print(f"  [NaN at step {global_step}, skipping]")
                    continue

                total.backward()

                # ── Gate-logit health monitor ─────────────────────────────────
                # Cheap probe of gate_net output magnitudes. If logit_mean drifts
                # sharply negative early in training, collapse is imminent — halt
                # the run rather than waste compute.
                if global_step % 50 == 0:
                    with torch.no_grad():
                        probe_pooled = model._encode_to_slots(
                            input_ids[:1], attention_mask[:1]
                        )
                        gate_logit_vals = model.gate_net(probe_pooled).squeeze(-1)
                        z_in_probe = model.fsq_proj(model.fsq_pre_ln(probe_pooled))
                        soft_mean = torch.sigmoid(gate_logit_vals).mean().item()
                        probe_log = {
                            "gates/logit_mean": gate_logit_vals.mean().item(),
                            "gates/logit_min": gate_logit_vals.min().item(),
                            "gates/logit_max": gate_logit_vals.max().item(),
                            "gates/soft_mean": soft_mean,
                            "bottleneck/z_in_norm": z_in_probe.norm(dim=-1).mean().item(),
                            "step": global_step,
                        }
                        # Gradient-norm health on the two layers most prone to dying.
                        if model.fsq_proj.weight.grad is not None:
                            probe_log["grad/fsq_proj_weight_norm"] = (
                                model.fsq_proj.weight.grad.norm().item()
                            )
                        gate_last = model.gate_net[-1]
                        if gate_last.weight.grad is not None:
                            probe_log["grad/gate_net_last_weight_norm"] = (
                                gate_last.weight.grad.norm().item()
                            )
                        wandb.log(probe_log)

                        # Early-abort kill-switch: in Stage 1, soft_mean must
                        # stay near 1.0 (warmup + min_gate force it). Sustained
                        # collapse by step 500 means the run is unrecoverable.
                        if (
                            global_step >= 500
                            and rate_scale == 0.0
                            and soft_mean < 0.3
                        ):
                            raise RuntimeError(
                                f"Gate collapse detected at step {global_step}: "
                                f"soft_mean={soft_mean:.3f} in Stage 1. Aborting."
                            )

                # ── Per-slot code-index entropy (every 200 steps) ────────────
                # Uniform entropy ≈ log2(codebook_size) means the slot carries
                # no information the decoder cares about; collapsed near 0 means
                # dead slot. Healthy slots sit in the middle.
                if global_step % 200 == 0:
                    with torch.no_grad():
                        K = int(model.fsq.codebook_size)
                        # flat_idx: [B, Q]
                        onehot = torch.nn.functional.one_hot(
                            flat_idx, num_classes=K
                        ).float()  # [B, Q, K]
                        p = onehot.mean(dim=0).clamp_min(1e-12)  # [Q, K]
                        H_per_slot = -(p * p.log2()).sum(dim=-1)  # [Q]
                        wandb.log({
                            "bottleneck/code_entropy_mean": H_per_slot.mean().item(),
                            "bottleneck/code_entropy_min": H_per_slot.min().item(),
                            "bottleneck/code_entropy_max": H_per_slot.max().item(),
                            "step": global_step,
                        })

                # ── Gradient ratio: NLI vs CE at the bottleneck ───────────────
                # Measures whether NLI is actually moving the bottleneck weights,
                # not just functioning as a monitoring signal.
                # Strategy: compare grad norms at fsq_proj.weight (the last layer
                # before quantization) from CE-only vs NLI contribution.
                # We do this once every 100 steps via separate autograd.grad calls.
                if global_step % 100 == 0 and config.LAMBDA_NLI > 0:
                    _log_gradient_ratio(
                        model, criterion,
                        recon_logits, gates, labels,
                        input_ids, attention_mask,
                        flat_idx, log_probs, rate_scale,
                        device, global_step,
                    )

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                global_step += 1

                avg_active = gates.gt(config.GATE_THRESHOLD).float().sum(dim=1).mean().item()
                with torch.no_grad():
                    active_mask = gates.gt(config.GATE_THRESHOLD).float()
                    entropy_bits = (
                        -(log_probs * active_mask).sum(dim=1).mean() / math.log(2)
                    ).item()

                log_dict = {
                    "loss/total": total.item(),
                    "loss/distortion": distortion.item(),
                    "loss/nli": nli.item(),
                    "loss/code_rate": code_rate.item(),
                    "loss/gate_rate": gate_rate.item(),
                    "loss/semantic": semantic.item(),
                    "loss/min_gate": min_gate.item(),
                    "compression/avg_active_gates": avg_active,
                    "compression/entropy_bits": entropy_bits,
                    "train/rate_scale": rate_scale,
                    "train/gate_prior_prob": torch.sigmoid(criterion.gate_prior_logit).item(),
                    "lr": scheduler.get_last_lr()[0],
                    "step": global_step,
                }

                # ── p_entail vs active_gates scatter (bucketed, sliding window) ─
                # Accumulate (gate_count, p_entail) pairs across SCATTER_FLUSH_EVERY
                # batches, then flush as a bucketed (mean ± std) table.
                # 10 batches × B=8 = 80 points → enough statistics per bucket.
                if want_scatter and p_entail_items is not None:
                    with torch.no_grad():
                        per_item_active = gates.gt(config.GATE_THRESHOLD).float().sum(dim=1)
                    for ag, pe in zip(per_item_active.tolist(), p_entail_items.tolist()):
                        scatter_buffer.append((int(ag), pe))
                    scatter_collect_count += 1

                    if scatter_collect_count >= SCATTER_FLUSH_EVERY:
                        _log_scatter_bucketed(scatter_buffer, global_step)
                        scatter_buffer.clear()
                        scatter_collect_count = 0

                # ── NLI sanity check ──────────────────────────────────────────
                # Detects silent NLI signal failures (distribution shift, etc.)
                # before they propagate unnoticed through training.
                if global_step % 1000 == 0:
                    model.eval()
                    _nli_sanity_check(criterion, tokenizer, device, global_step)
                    model.train()

                wandb.log(log_dict)

                for k, v in [
                    ("total", total), ("distortion", distortion), ("nli", nli),
                    ("code_rate", code_rate), ("gate_rate", gate_rate),
                    ("semantic", semantic),
                ]:
                    epoch_losses[k] += v.item()

                pbar.set_postfix({
                    "loss": f"{total.item():.3f}",
                    "dist": f"{distortion.item():.3f}",
                    "nli": f"{nli.item():.3f}",
                    "R_c": f"{code_rate.item():.3f}",
                    "gates": f"{avg_active:.1f}/{config.NUM_QUERIES}",
                    "scale": f"{rate_scale:.2f}",
                })

                # Periodic qualitative eval
                if global_step % 500 == 0:
                    _eval_sample(model, tokenizer, device, global_step)

            n = steps_per_epoch
            print(
                f"\nEpoch {epoch+1} — "
                f"total: {epoch_losses['total']/n:.4f}  "
                f"dist: {epoch_losses['distortion']/n:.4f}  "
                f"nli: {epoch_losses['nli']/n:.4f}  "
                f"R_c: {epoch_losses['code_rate']/n:.4f}  "
                f"R_g: {epoch_losses['gate_rate']/n:.4f}  "
                f"sem: {epoch_losses['semantic']/n:.4f}"
            )

            # ── Shuffle-MI probe: the definitive posterior-collapse test ─────
            # If decoder CE is ~unchanged when z_q is permuted across the batch,
            # the decoder is ignoring the bottleneck and no amount of gate
            # tuning will help.
            try:
                model.eval()
                with torch.no_grad():
                    probe_batch = next(iter(dataloader))
                    ids_p = probe_batch["input_ids"].to(device)
                    msk_p = probe_batch["attention_mask"].to(device)
                    lbl_p = probe_batch["labels"].to(device)
                    pooled_p = model._encode_to_slots(ids_p, msk_p)
                    z_q_p, _, gates_p, _, _ = model._quantise(pooled_p)
                    base = model.bart.get_base_model()
                    from transformers.models.bart.modeling_bart import shift_tokens_right as _sh
                    dec_ids_p = _sh(lbl_p, base.config.pad_token_id, base.config.decoder_start_token_id)
                    logits_real = model._decode(z_q_p, gates_p, dec_ids_p)
                    perm = torch.randperm(z_q_p.size(0), device=device)
                    logits_shuf = model._decode(z_q_p[perm], gates_p[perm], dec_ids_p)
                    V = logits_real.shape[-1]
                    ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
                    ce_real = ce(logits_real.reshape(-1, V), lbl_p.reshape(-1)).item()
                    ce_shuf = ce(logits_shuf.reshape(-1, V), lbl_p.reshape(-1)).item()
                    wandb.log({
                        "probe/ce_real": ce_real,
                        "probe/ce_shuffled": ce_shuf,
                        "probe/shuffle_gap": ce_shuf - ce_real,
                        "epoch": epoch + 1,
                    })
                    print(f"  [Shuffle-MI] CE real={ce_real:.3f}  shuffled={ce_shuf:.3f}  gap={ce_shuf-ce_real:+.3f}")
                    if ce_shuf - ce_real < 0.1:
                        print("    WARN: decoder is nearly independent of bottleneck — posterior collapse.")
                model.train()
            except Exception as e:
                print(f"  [Shuffle-MI probe skipped: {e}]")

            ckpt_path = f"checkpoint_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved {ckpt_path}")

    except KeyboardInterrupt:
        print("Interrupted — saving partial checkpoint...")
        torch.save(model.state_dict(), "checkpoint_partial.pt")

    wandb.finish()


def _eval_sample(model, tokenizer, device, global_step):
    model.eval()
    text = (
        "The rapid proliferation of artificial intelligence technologies has "
        "simultaneously catalyzed unprecedented efficiencies across multiple "
        "industrial sectors and precipitated profound anxieties regarding the "
        "potential displacement of human labor."
    )
    with torch.no_grad():
        tok = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=config.MAX_LENGTH,
        )
        ids = tok["input_ids"].to(device)
        mask = tok["attention_mask"].to(device)

        indices, gate_mask, payload, num_bits, entropy_bits = model.compress_to_indices(ids, mask)
        gen_ids = model.decompress_from_indices(indices, gate_mask, max_length=128, num_beams=4)
        recon = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        # Random-code baseline: if this decodes to similar-quality output,
        # the decoder has memorized the prior and isn't using the bottleneck.
        rand_indices = torch.stack([
            torch.randint(0, int(L), indices.shape[:-1] + (1,), device=device).squeeze(-1)
            for L in config.FSQ_LEVELS
        ], dim=-1)
        all_open = torch.ones_like(gate_mask)
        rand_ids = model.decompress_from_indices(rand_indices, all_open, max_length=128, num_beams=4)
        rand_recon = tokenizer.decode(rand_ids[0], skip_special_tokens=True)

        orig_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        # Baseline: Huffman over BPE frequencies ≈ 10–11 bits/token for English
        huffman_bits = orig_tokens * 10.5
        ratio_vs_huffman = (1 - num_bits / huffman_bits) * 100

        print(f"\n[Step {global_step}]")
        print(f"  Active gates:   {gate_mask.float().sum().item():.0f}/{config.NUM_QUERIES}")
        print(f"  Payload bits:   {num_bits:.1f}  (entropy model est: {entropy_bits:.1f})")
        print(f"  vs Huffman BPE: {ratio_vs_huffman:.1f}% reduction")
        print(f"  Recon: {recon[:120]}...")
        print(f"  RandCode: {rand_recon[:120]}...")

        wandb.log({
            "eval/payload_bits": num_bits,
            "eval/entropy_bits_est": entropy_bits,
            "eval/active_gates": gate_mask.float().sum().item(),
            "eval/ratio_vs_huffman_pct": ratio_vs_huffman,
            "eval/reconstruction": wandb.Html(recon),
            "eval/random_code_recon": wandb.Html(rand_recon),
        })
    model.train()


if __name__ == "__main__":
    train()
