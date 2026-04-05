import torch
from model import SemanticAutoencoder
from transformers import BartTokenizer

device = torch.device("mps")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = SemanticAutoencoder().to(device)
model.load_state_dict(torch.load("seq2seq_simplification_epoch_3.pt", map_location=device), strict=False)
model.eval()

text = "The rapid proliferation of artificial intelligence technologies has simultaneously catalyzed unprecedented efficiencies across multiple industrial sectors and precipitated profound anxieties regarding the potential displacement of human labor."
inputs = tokenizer(text, return_tensors="pt", truncation=True)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

base_model = model.bart.get_base_model()
encoder = base_model.get_encoder()
encoder_outputs = encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
hidden_states = encoder_outputs.last_hidden_state

batch_size = hidden_states.size(0)
queries = model.query_tokens.expand(batch_size, -1, -1).contiguous()
key_padding_mask = (attention_mask == 0) if attention_mask is not None else None

pooled_states, _ = model.cross_attention(
    query=queries,
    key=hidden_states.contiguous(),
    value=hidden_states.contiguous(),
    key_padding_mask=key_padding_mask
)

z_logits = base_model.lm_head(pooled_states)
z_token_ids = z_logits.argmax(dim=-1)

print("z_token_ids:", z_token_ids)
