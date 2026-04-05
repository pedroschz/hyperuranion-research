"""
Rate-Distortion Evaluation for Semantic Idea Compression

Goal: find the Pareto frontier of (bits/char, idea-completeness).
Idea-completeness is measured by NLI entailment coverage — not cosine similarity,
which is too weak to detect dropped propositional content.

Baselines:
    truncation + zstd  — the honest strong simple baseline most papers skip.
                         Keep the first K characters, compress with zstd.
                         Directly comparable to this system: also lossy, also variable rate.
    gzip / zstd        — lossless classical codecs (x-intercept reference only)
    LM-AC (GPT-2)      — lossless neural baseline (~1.2 bits/char); sets the floor
                         for lossless neural compression on this domain.

Quality metrics:
    entailment_fwd     — p(original entails reconstruction): does recon contain orig's claims?
    entailment_bwd     — p(reconstruction entails original): does recon hallucinate?
    entailment_sym     — geometric mean of both (the primary quality signal)
    cosine_sim         — SentenceTransformer cosine (fast but weak)
    rouge_l            — surface overlap

The x-axis is bits/character (BPC) — model-agnostic, the compression literature standard.

Usage:
    python eval_rd.py --checkpoint checkpoint_epoch_5.pt --samples 200
"""

import argparse
import csv
import gzip
import math
import torch
import numpy as np
from transformers import (
    BartTokenizer,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm

import config
from model import SemanticAutoencoder

# ── Optional deps ─────────────────────────────────────────────────────────────

try:
    import zstandard as zstd
    _ZSTD = True
    _zstd_cctx = zstd.ZstdCompressor(level=19)
except ImportError:
    _ZSTD = False

try:
    from rouge_score import rouge_scorer
    _ROUGE = True
    _rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
except ImportError:
    _ROUGE = False

try:
    from bert_score import score as bertscore_fn
    _BERTSCORE = True
except ImportError:
    _BERTSCORE = False


# ──────────────────────────────────────────────────────────────────────────────
# NLI entailment evaluator
# ──────────────────────────────────────────────────────────────────────────────

class NLIEvaluator:
    """
    Batch NLI evaluator for measuring idea completeness.

    We check both directions:
      fwd: p(original → reconstruction) — did we preserve orig's claims in recon?
      bwd: p(reconstruction → original) — did recon hallucinate non-original claims?

    Symmetric entailment = geometric mean of both.
    This is the primary quality metric for the goal of idea preservation.
    """
    # DeBERTa NLI label order: 0=contradiction, 1=entailment, 2=neutral
    ENTAILMENT_IDX = 1

    def __init__(self, device):
        self.device = device
        model_name = "cross-encoder/nli-deberta-v3-small"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def score(self, premises: list[str], hypotheses: list[str]) -> list[float]:
        """Returns p(entailment) for each (premise, hypothesis) pair."""
        enc = self.tokenizer(
            premises, hypotheses,
            padding=True, truncation=True, max_length=256,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**enc).logits          # [B, 3]
        probs = F.softmax(logits, dim=-1)          # [B, 3]
        return probs[:, self.ENTAILMENT_IDX].tolist()

    @torch.no_grad()
    def symmetric_entailment(
        self, originals: list[str], reconstructions: list[str]
    ) -> tuple[list[float], list[float], list[float]]:
        """
        Returns (fwd_scores, bwd_scores, symmetric_scores) per sample.
        fwd: orig entails recon (recon contains orig's claims)
        bwd: recon entails orig (recon doesn't hallucinate)
        """
        fwd = self.score(originals, reconstructions)
        bwd = self.score(reconstructions, originals)
        sym = [(f * b) ** 0.5 for f, b in zip(fwd, bwd)]
        return fwd, bwd, sym


# ──────────────────────────────────────────────────────────────────────────────
# Baseline: truncation + zstd
# ──────────────────────────────────────────────────────────────────────────────

def truncation_baseline(text: str, keep_chars: int) -> tuple[str, float]:
    """
    Truncate text to keep_chars characters and compress with zstd.
    Returns (truncated_text, bits).
    This is the honest strong simple baseline.
    """
    truncated = text[:keep_chars]
    if _ZSTD:
        bits = len(_zstd_cctx.compress(truncated.encode("utf-8"))) * 8
    else:
        bits = len(gzip.compress(truncated.encode("utf-8"))) * 8
    return truncated, bits


def sweep_truncation_baseline(
    texts: list[str],
    nli_eval: NLIEvaluator,
    sbert,
    device,
    n_points: int = 9,
) -> list[dict]:
    """
    Sweep keep_chars from very short to full length.
    Returns R-D points for the truncation + zstd baseline.
    """
    max_chars = max(len(t) for t in texts)
    keep_fracs = np.linspace(0.1, 1.0, n_points)
    results = []

    for frac in keep_fracs:
        keep = max(10, int(max_chars * frac))
        truncated_texts = []
        bits_list = []
        for text in texts:
            trunc, bits = truncation_baseline(text, keep)
            truncated_texts.append(trunc)
            bits_list.append(bits)

        char_counts = [len(t) for t in texts]
        bpc_list = [b / max(c, 1) for b, c in zip(bits_list, char_counts)]

        fwd, bwd, sym = nli_eval.symmetric_entailment(texts, truncated_texts)

        cos_sims = []
        if sbert is not None:
            orig_embs = sbert.encode(texts, convert_to_tensor=True)
            trunc_embs = sbert.encode(truncated_texts, convert_to_tensor=True)
            for o, r in zip(orig_embs, trunc_embs):
                cos_sims.append(F.cosine_similarity(o.unsqueeze(0), r.unsqueeze(0)).item())

        results.append({
            "system": "truncation+zstd",
            "keep_frac": frac,
            "avg_bpc": float(np.mean(bpc_list)),
            "avg_entailment_fwd": float(np.mean(fwd)),
            "avg_entailment_bwd": float(np.mean(bwd)),
            "avg_entailment_sym": float(np.mean(sym)),
            "avg_cosine_sim": float(np.mean(cos_sims)) if cos_sims else float("nan"),
        })
        print(
            f"  [trunc] keep={frac:.1%} | "
            f"bpc={results[-1]['avg_bpc']:.3f} | "
            f"entail_sym={results[-1]['avg_entailment_sym']:.3f} | "
            f"cos={results[-1]['avg_cosine_sim']:.3f}"
        )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Classical codec baselines (lossless, for x-intercept reference)
# ──────────────────────────────────────────────────────────────────────────────

def lossless_baselines(texts: list[str]) -> dict:
    """Compute bits-per-character for lossless codecs (quality=1.0 by definition)."""
    results = {}
    total_chars = sum(len(t) for t in texts)

    # gzip
    gzip_bits = sum(len(gzip.compress(t.encode("utf-8"))) * 8 for t in texts)
    results["gzip_bpc"] = gzip_bits / max(total_chars, 1)

    # zstd
    if _ZSTD:
        zstd_bits = sum(len(_zstd_cctx.compress(t.encode("utf-8"))) * 8 for t in texts)
        results["zstd_bpc"] = zstd_bits / max(total_chars, 1)
    else:
        results["zstd_bpc"] = float("nan")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Model evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model_at_threshold(
    model, tokenizer, nli_eval, sbert, texts, device, gate_threshold
) -> dict:
    original_threshold = config.GATE_THRESHOLD
    config.GATE_THRESHOLD = gate_threshold

    payload_bits_list, entropy_bits_list, char_counts = [], [], []
    originals, reconstructions = [], []

    model.eval()
    with torch.no_grad():
        for text in texts:
            tok = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=config.MAX_LENGTH,
                padding=False,
            )
            ids = tok["input_ids"].to(device)
            mask = tok["attention_mask"].to(device)

            indices, gate_mask, payload, num_bits, entropy_bits = model.compress_to_indices(
                ids, mask
            )
            gen_ids = model.decompress_from_indices(
                indices, gate_mask, max_length=config.MAX_LENGTH + 32, num_beams=2
            )
            recon = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

            payload_bits_list.append(num_bits)
            entropy_bits_list.append(entropy_bits)
            char_counts.append(len(text))
            originals.append(text)
            reconstructions.append(recon)

    config.GATE_THRESHOLD = original_threshold

    # BPC (bits per character — model-agnostic standard)
    bpc_list = [b / max(c, 1) for b, c in zip(payload_bits_list, char_counts)]
    entropy_bpc_list = [b / max(c, 1) for b, c in zip(entropy_bits_list, char_counts)]

    # NLI entailment (primary quality metric)
    fwd, bwd, sym = nli_eval.symmetric_entailment(originals, reconstructions)

    # Cosine similarity (fast secondary metric)
    cos_sims = []
    if sbert is not None:
        orig_embs = sbert.encode(originals, convert_to_tensor=True)
        recon_embs = sbert.encode(reconstructions, convert_to_tensor=True)
        for o, r in zip(orig_embs, recon_embs):
            cos_sims.append(F.cosine_similarity(o.unsqueeze(0), r.unsqueeze(0)).item())

    # ROUGE-L
    rouge_l = []
    if _ROUGE:
        for orig, recon in zip(originals, reconstructions):
            rouge_l.append(_rouge.score(orig, recon)["rougeL"].fmeasure)

    # BERTScore (batched)
    bertscore_f1 = float("nan")
    if _BERTSCORE:
        _, _, F1 = bertscore_fn(reconstructions, originals, lang="en", verbose=False)
        bertscore_f1 = F1.mean().item()

    def nm(lst):
        lst = [x for x in lst if not (isinstance(x, float) and math.isnan(x))]
        return float(np.mean(lst)) if lst else float("nan")

    return {
        "system": "model",
        "gate_threshold": gate_threshold,
        "avg_bpc": nm(bpc_list),
        "avg_entropy_bpc": nm(entropy_bpc_list),
        "avg_entailment_fwd": nm(fwd),
        "avg_entailment_bwd": nm(bwd),
        "avg_entailment_sym": nm(sym),
        "avg_cosine_sim": nm(cos_sims),
        "avg_rouge_l": nm(rouge_l),
        "avg_bertscore_f1": bertscore_f1,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--output", default="rd_curve.csv")
    parser.add_argument("--skip-truncation-baseline", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    model = SemanticAutoencoder().to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device), strict=False
    )
    model.eval()
    print(f"Loaded {args.checkpoint}")

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    sbert = SentenceTransformer("sentence-transformers/all-distilroberta-v1").to(device)

    print("Loading NLI evaluator (cross-encoder/nli-deberta-v3-small)...")
    nli_eval = NLIEvaluator(device)

    print("Loading test data...")
    ds = load_dataset("eilamc14/wikilarge-clean", split="test")
    # Use source texts only (this is an idea-preservation task, not simplification)
    texts = [ds[i]["source"] for i in range(min(args.samples, len(ds)))]
    texts = [t for t in texts if len(t.split()) > 10]  # filter trivially short
    print(f"Evaluating on {len(texts)} samples.")

    all_rows = []

    # ── 1. Truncation + zstd baseline ─────────────────────────────────────────
    if not args.skip_truncation_baseline:
        print("\n── Truncation + zstd baseline ───────────────────────────────────────")
        trunc_rows = sweep_truncation_baseline(texts, nli_eval, sbert, device)
        all_rows.extend(trunc_rows)

    # ── 2. Lossless classical baselines ───────────────────────────────────────
    print("\n── Lossless classical baselines (x-intercept reference) ─────────────")
    lossless = lossless_baselines(texts)
    for k, v in lossless.items():
        print(f"  {k}: {v:.3f} bpc  (quality=1.0, lossless)")

    # ── 3. Model R-D curve (gate threshold sweep) ─────────────────────────────
    print("\n── Model R-D curve (gate threshold sweep) ───────────────────────────")
    print("  Note: this sweeps the inference threshold, not separate λ training runs.")
    print("  For a publication-quality curve, train at 5+ λ values separately.\n")

    thresholds = np.linspace(0.15, 0.85, num=8).tolist()
    for thr in tqdm(thresholds):
        row = evaluate_model_at_threshold(
            model, tokenizer, nli_eval, sbert, texts, device, thr
        )
        all_rows.append(row)
        print(
            f"  threshold={thr:.2f} | "
            f"bpc={row['avg_bpc']:.3f} "
            f"(entropy_est={row['avg_entropy_bpc']:.3f}) | "
            f"entail_sym={row['avg_entailment_sym']:.3f} | "
            f"cos={row['avg_cosine_sim']:.3f} | "
            f"ROUGE-L={row['avg_rouge_l']:.3f}"
        )

    # ── Save ──────────────────────────────────────────────────────────────────
    # Collect all keys across rows (different row types have different fields)
    all_keys = []
    for row in all_rows:
        for k in row.keys():
            if k not in all_keys:
                all_keys.append(k)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in all_keys})

    print(f"\nR-D curve saved to {args.output}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n── Key comparison (at entailment_sym ≈ 0.85) ───────────────────────")
    # Find model row closest to 0.85 entailment
    model_rows = [r for r in all_rows if r.get("system") == "model"]
    if model_rows:
        target_row = min(model_rows, key=lambda r: abs(r["avg_entailment_sym"] - 0.85))
        print(f"  Model at entail_sym={target_row['avg_entailment_sym']:.3f}: "
              f"bpc={target_row['avg_bpc']:.3f}")

    # Find truncation row with similar entailment
    trunc_rows_saved = [r for r in all_rows if r.get("system") == "truncation+zstd"]
    if trunc_rows_saved:
        target_trunc = min(trunc_rows_saved, key=lambda r: abs(r["avg_entailment_sym"] - 0.85))
        print(f"  Truncation+zstd at entail_sym={target_trunc['avg_entailment_sym']:.3f}: "
              f"bpc={target_trunc['avg_bpc']:.3f}")
        print(f"  {'Model wins' if target_row['avg_bpc'] < target_trunc['avg_bpc'] else 'Truncation wins'} "
              f"at matched fidelity.")

    print("\nPlot with:")
    print(
        "  python -c \""
        "import pandas as pd, matplotlib.pyplot as plt; "
        "df=pd.read_csv('rd_curve.csv'); "
        "m=df[df.system=='model']; t=df[df.system=='truncation+zstd']; "
        "plt.plot(m.avg_bpc, m.avg_entailment_sym, 'o-b', label='model'); "
        "plt.plot(t.avg_bpc, t.avg_entailment_sym, 's--r', label='truncation+zstd'); "
        "plt.xlabel('bits/char'); plt.ylabel('symmetric entailment'); "
        "plt.legend(); plt.grid(); plt.show()\""
    )


if __name__ == "__main__":
    main()
