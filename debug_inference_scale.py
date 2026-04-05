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
    pooled = model._encode_to_slots(ids, mask)
    z_q, z_qh, gates, flat_idx = model._quantise(pooled)
    
    print("Predicted gates:", gates.squeeze().tolist())
    
    # Simulate decoder exactly as in training
    decoder_input_ids = torch.tensor([[model.bart.get_base_model().config.decoder_start_token_id]]).to(device)
    
    # Let's generate tokens manually scaling by EXACT predicted gates
    enc_hidden = model.fsq_unproj(z_q)
    enc_hidden_scaled = enc_hidden * gates.unsqueeze(-1)
    
    enc_mask = torch.ones_like(gates, dtype=torch.long)
    enc_outputs = type('obj', (object,), {'last_hidden_state': enc_hidden_scaled})()
    
    gen_ids = model.bart.get_base_model().generate(
        encoder_outputs=enc_outputs,
        attention_mask=enc_mask,
        max_length=64,
        num_beams=2,
    )
    
    print("Recon with EXACT gates:", tokenizer.decode(gen_ids[0], skip_special_tokens=True))
    
    # Now generate scaling by 1.0 (what eval_rd does)
    enc_hidden_1 = enc_hidden * torch.ones_like(gates).unsqueeze(-1)
    enc_outputs_1 = type('obj', (object,), {'last_hidden_state': enc_hidden_1})()
    gen_ids_1 = model.bart.get_base_model().generate(
        encoder_outputs=enc_outputs_1,
        attention_mask=enc_mask,
        max_length=64,
        num_beams=2,
    )
    
    print("Recon with 1.0 gates:", tokenizer.decode(gen_ids_1[0], skip_special_tokens=True))

