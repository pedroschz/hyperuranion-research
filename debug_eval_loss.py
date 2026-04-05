import torch
import config
from model import SemanticAutoencoder
from transformers import BartTokenizer
from loss import RateDistortionLoss
import math

device = "mps"
model = SemanticAutoencoder().to(device)
model.load_state_dict(torch.load("checkpoint_epoch_5.pt", map_location=device), strict=False)
model.eval()

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
criterion = RateDistortionLoss(model.bart.get_base_model().config.vocab_size).to(device)

text = "The rapid proliferation of artificial intelligence technologies has simultaneously catalyzed unprecedented efficiencies across multiple industrial sectors and precipitated profound anxieties regarding the potential displacement of human labor."
tok = tokenizer(text, return_tensors="pt", truncation=True, max_length=config.MAX_LENGTH, padding="max_length")
ids = tok["input_ids"].to(device)
mask = tok["attention_mask"].to(device)
labels = ids.clone()
labels[labels == tokenizer.pad_token_id] = -100

with torch.no_grad():
    recon_logits, gates, z_qh, flat_idx, log_probs = model(ids, mask, labels=labels, rate_scale=1.0)
    print("Gates (mean):", gates.mean().item())
    
    loss_dict = criterion(
        recon_logits=recon_logits,
        target_ids=labels,
        gates=gates,
        gate_mask=(gates > config.GATE_THRESHOLD),
        log_probs=log_probs,
        rate_scale=1.0,
        global_step=10000,
        premise_ids=ids,
        premise_mask=mask,
        recon_ids=recon_logits.argmax(dim=-1)
    )
    for k, v in loss_dict.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.item():.4f}")
        else:
            print(f"{k}: {v}")

