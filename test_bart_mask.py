import torch
from transformers import BartModel

model = BartModel.from_pretrained("facebook/bart-base")
B, T_enc, T_dec, H = 2, 5, 4, 768

enc_hidden = torch.randn(B, T_enc, H)
enc_mask = torch.zeros(B, T_enc, dtype=torch.long) # ALL ZEROS

dec_input = torch.randint(0, 1000, (B, T_dec))

out = model.decoder(
    input_ids=dec_input,
    encoder_hidden_states=enc_hidden,
    encoder_attention_mask=enc_mask,
)

print(out.last_hidden_state.isnan().any())
print(out.last_hidden_state.shape)
