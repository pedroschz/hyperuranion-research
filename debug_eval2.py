import torch
import config
from model import SemanticAutoencoder
from transformers import BartTokenizer
import torchac

device = "mps"
model = SemanticAutoencoder().to(device)
model.load_state_dict(torch.load("checkpoint_epoch_4.pt", map_location=device), strict=False)
model.eval()

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
text = "The rapid proliferation of artificial intelligence technologies has simultaneously catalyzed unprecedented efficiencies across multiple industrial sectors and precipitated profound anxieties regarding the potential displacement of human labor."
tok = tokenizer(text, return_tensors="pt", truncation=True, max_length=config.MAX_LENGTH, padding=False)
ids = tok["input_ids"].to(device)
mask = tok["attention_mask"].to(device)

with torch.no_grad():
    indices, gate_mask, payload, num_bits, entropy_bits = model.compress_to_indices(ids, mask)
    
    flat_idx = model.fsq.to_flat_index(indices)
    
    # Test torchac on whole array
    cdfs = model.entropy_model.teacher_forced_cdfs(flat_idx)
    sym_t = flat_idx.to(torch.int16).cpu()
    
    ac_payload = torchac.encode_float_cdf(cdfs.cpu(), sym_t)
    print("New payload bytes:", len(ac_payload))
    print("New payload bits:", len(ac_payload) * 8)
    print("Entropy bits:", entropy_bits)
