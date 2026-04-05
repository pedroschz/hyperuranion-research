import sys
import torch
print("Loading model module...")
sys.stdout.flush()

from model import SemanticAutoencoder

def test():
    print("Setting device...")
    sys.stdout.flush()
    device = torch.device("mps")
    
    print("Init model...")
    sys.stdout.flush()
    model = SemanticAutoencoder()
    
    print("Moving model to device...")
    sys.stdout.flush()
    model = model.to(device)
    
    print("Creating tensors...")
    sys.stdout.flush()
    input_ids = torch.randint(0, 100, (2, 32)).to(device)
    attention_mask = torch.ones((2, 32)).to(device)
    labels = torch.randint(0, 100, (2, 32)).to(device)
    
    print("1. Encoder forward")
    sys.stdout.flush()
    base_model = model.bart.get_base_model()
    encoder = base_model.get_encoder()
    encoder_outputs = encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    hidden_states = encoder_outputs.last_hidden_state.contiguous()
    
    print("2. Setup queries")
    sys.stdout.flush()
    batch_size = hidden_states.size(0)
    queries = model.query_tokens.expand(batch_size, -1, -1).contiguous()
    key_padding_mask = None
    
    print("3. Cross attention forward")
    sys.stdout.flush()
    pooled_states, _ = model.cross_attention(
        query=queries,
        key=hidden_states,
        value=hidden_states,
        key_padding_mask=key_padding_mask
    )
    
    print("4. LM Head")
    sys.stdout.flush()
    z_logits = base_model.lm_head(pooled_states)
    
    print("Success!")
    sys.stdout.flush()
    sys.exit(0)

if __name__ == "__main__":
    test()
