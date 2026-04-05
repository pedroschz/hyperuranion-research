import torch
from sentence_transformers import SentenceTransformer
from transformers import BartTokenizer
from model import SemanticAutoencoder
import config


def compress_and_decompress(text, model_path=None):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model = SemanticAutoencoder().to(device)

    if model_path:
        model.load_state_dict(
            torch.load(model_path, map_location=device), strict=False
        )
    model.eval()

    sbert = SentenceTransformer("sentence-transformers/all-distilroberta-v1").to(device)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=config.MAX_LENGTH,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        # ── Compress ──────────────────────────────────────────────────────────
        indices, gate_mask, used_bits = model.compress(input_ids, attention_mask)

        active_slots = gate_mask.float().sum().item()
        orig_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        orig_bits = orig_tokens * 16
        compression_pct = (1 - used_bits / orig_bits) * 100 if orig_bits > 0 else 0

        # ── Decompress ────────────────────────────────────────────────────────
        gen_ids = model.decompress(
            indices, gate_mask, max_length=128, num_beams=4, early_stopping=True
        )
        recon = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        # ── Semantic fidelity ─────────────────────────────────────────────────
        orig_emb = sbert.encode(text, convert_to_tensor=True)
        rec_emb = sbert.encode(recon, convert_to_tensor=True)
        cos_sim = torch.nn.functional.cosine_similarity(
            orig_emb.unsqueeze(0), rec_emb.unsqueeze(0)
        ).item()

    print("\n── Original ──────────────────────────────────────────────────────")
    print(text)
    print("\n── Reconstruction ────────────────────────────────────────────────")
    print(recon)
    print("\n── Compression Stats ─────────────────────────────────────────────")
    print(f"  Original:       {orig_tokens} tokens → {orig_bits} bits")
    print(f"  Compressed:     {active_slots:.0f}/{config.NUM_QUERIES} active slots "
          f"→ {used_bits:.1f} bits")
    print(f"  Compression:    {compression_pct:.1f}%")
    print(f"  FSQ config:     levels={config.FSQ_LEVELS}, "
          f"{model.fsq.bits_per_code:.2f} bits/code")
    print(f"\n── Semantic Fidelity ─────────────────────────────────────────────")
    print(f"  Cosine similarity: {cos_sim:.4f}")


if __name__ == "__main__":
    test_cases = [
        "The rapid proliferation of artificial intelligence technologies has simultaneously catalyzed unprecedented efficiencies across multiple industrial sectors and precipitated profound anxieties regarding the potential displacement of human labor.",
        "I love humanity.",
        "Scientists have discovered that regular exercise significantly reduces the risk of cardiovascular disease, diabetes, and certain forms of cancer while also improving mental health outcomes.",
    ]

    for text in test_cases:
        compress_and_decompress(text, model_path="checkpoint_epoch_3.pt")
        print("\n" + "=" * 70 + "\n")
