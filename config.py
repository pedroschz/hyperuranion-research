# ─── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 8               # reduced from 16: longer sequences + larger bottleneck
MAX_LENGTH = 128             # up from 64: arguments/reasoning chains span multiple sentences
LEARNING_RATE = 2e-4         # 2e-4 is needed for randomly initialized Perceiver/FSQ to learn
NUM_EPOCHS = 5

# ─── LoRA ───────────────────────────────────────────────────────────────────────
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# ─── Perceiver bottleneck ───────────────────────────────────────────────────────
# Sizing rationale for semantic idea compression (goal: preserve all propositional content):
#
#   Input:   128 tokens × ~5 bits/token (LM entropy of English prose) ≈ 640 bits of ideas
#   Bottleneck target: capture ≥ 80% of that → need ≥ 512 bits
#
#   NUM_QUERIES × bits_per_code = 32 × 12.0 = 384 bits  ← aggressive but workable
#   NUM_QUERIES × bits_per_code = 48 × 12.0 = 576 bits  ← near-lossless target
#
# Start with 32 queries. If reconstruction quality is insufficient (entailment < 0.85),
# increase to 48. The gate network will naturally prune queries that aren't needed,
# so over-provisioning is safe — it just increases the maximum rate, not the typical rate.
NUM_QUERIES = 32
PERCEIVER_ITERS = 3

# ─── FSQ (Finite Scalar Quantization) ──────────────────────────────────────────
# [8, 8, 8, 8] → exactly 12.0 bits/code — uniform per dimension, easier to model.
# With 32 queries: up to 384 bits total.
# With the gate network: expected usage is 60–80% of slots → ~230–307 actual bits.
#
# Compare: gzip of 128 tokens of English prose ≈ 700–900 bits.
# Target: ≤ 400 bits while preserving all propositional content.
FSQ_LEVELS = [8, 8, 8, 8]
FSQ_DIMS = len(FSQ_LEVELS)

# ─── Entropy model ──────────────────────────────────────────────────────────────
ENTROPY_MODEL_DIM = 256
ENTROPY_MODEL_HEADS = 4
ENTROPY_MODEL_LAYERS = 3

# ─── Rate-Distortion ───────────────────────────────────────────────────────────
# λ_rate should be SMALL initially — faithful reconstruction of all ideas is
# the hard constraint; compression ratio is the objective to maximise subject to it.
# Increase λ_rate only after reconstruction quality (entailment ≥ 0.85) is established.
LAMBDA_RATE = 0.005          # lowered from 0.01: gentler ramp prevents snap collapse
LAMBDA_GATE = 0.005
LAMBDA_SEM = 0.0             # disabled: NLI strictly dominates CoSENT as quality signal,
                             # and removing SBERT (~66M params) saves ~1GB GPU memory.
                             # Re-enable only if NLI signal is noisy on your hardware.
LAMBDA_NLI = 0.1             # primary quality signal (differentiable via BART-large-mnli)
                             # Effective gradient magnitude at the bottleneck may be
                             # attenuated through MNLI's 12 encoder layers. Monitor
                             # "grad/nli_vs_ce_ratio_at_fsq" in wandb. If < 0.01,
                             # increase LAMBDA_NLI to compensate.

# ─── Gating ────────────────────────────────────────────────────────────────────
GATE_THRESHOLD = 0.5

# ─── Word Dropout (Decoder Input Masking) ──────────────────────────────────────
# Prevents posterior collapse via teacher-forcing loophole (Bowman et al. 2015).
# Randomly replaces a fraction of decoder input tokens with <mask>.
# The decoder cannot reconstruct masked positions from local context alone,
# so it is mathematically forced to attend to the bottleneck.
# 0.4 = 40% of input tokens are dropped (never the first BOS or padding tokens).
WORD_DROPOUT = 0.4

# ─── Training curriculum ───────────────────────────────────────────────────────
# Stage 1 [0, STAGE2_START):  rate off → model learns to encode all ideas
# Stage 2 [STAGE2_START, STAGE3_START): rate on → compress without losing ideas
# Stage 3 [STAGE3_START, NUM_EPOCHS): freeze backbone → refine entropy model
STAGE2_START = 2
STAGE3_START = 4
