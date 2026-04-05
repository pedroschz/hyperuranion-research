import torch
import config
from model import SemanticAutoencoder
from transformers import BartTokenizer

device = "mps"
model = SemanticAutoencoder().to(device)
model.load_state_dict(torch.load("checkpoint_epoch_5.pt", map_location=device), strict=False)
model.eval()

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
text = "The rapid proliferation of artificial intelligence technologies has simultaneously catalyzed unprecedented efficiencies across multiple industrial sectors and precipitated profound anxieties regarding the potential displacement of human labor."
tok = tokenizer(text, return_tensors="pt", truncation=True, max_length=config.MAX_LENGTH, padding=False)
ids = tok["input_ids"].to(device)
mask = tok["attention_mask"].to(device)

with torch.no_grad():
    indices, gate_mask, payload, num_bits, entropy_bits = model.compress_to_indices(ids, mask)
    print("Indices shape:", indices.shape)
    print("Gate mask:", gate_mask.int().tolist())
    print("Num bits:", num_bits)
    print("Entropy bits:", entropy_bits)
    print("Active gates:", gate_mask.float().sum().item())
    
    gen_ids = model.decompress_from_indices(indices, gate_mask, max_length=128, num_beams=2)
    recon = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    print("Recon:", recon)
