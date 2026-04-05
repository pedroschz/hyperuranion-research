import torch
import torch.optim as optim
import config
from model import SemanticAutoencoder
from transformers import BartTokenizer
from loss import RateDistortionLoss
import math

def test_overfit():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Initialize model and loss
    model = SemanticAutoencoder().to(device)
    model.train()
    
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    criterion = RateDistortionLoss(model.bart.get_base_model().config.vocab_size).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # 2. Create a single batch of complex sentences
    texts = [
        "The rapid proliferation of artificial intelligence technologies has simultaneously catalyzed unprecedented efficiencies across multiple industrial sectors and precipitated profound anxieties regarding the potential displacement of human labor.",
        "While classical economic theories posit that unregulated markets inevitably tend toward optimal resource allocation, empirical observations of recurring financial crises suggest the necessity of prudent regulatory frameworks.",
        "Although the initial phases of the software development lifecycle demand rigorous requirements analysis and architectural planning, agile methodologies emphasize iterative refinement and continuous stakeholder feedback.",
        "The intricate symbiotic relationships between diverse species within a thriving coral reef ecosystem demonstrate a fragile equilibrium that is acutely vulnerable to microscopic variations in ocean temperature and acidity."
    ]
    
    tok = tokenizer(texts, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
    input_ids = tok["input_ids"].to(device)
    attention_mask = tok["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    
    print("\nStarting local overfit test (200 steps)...")
    print("Steps 0-49: Stage 1 (Reconstruction only, rate_scale=0)")
    print("Steps 50-200: Stage 2 (Compression on, rate_scale=1.0)")
    print("-" * 100)
    print(f"{'Step':>4} | {'Loss':>6} | {'Dist':>6} | {'NLI':>6} | {'GateR':>6} | {'CodeR':>6} | {'Gates':>5} | {'Recon Example'}")
    print("-" * 100)
    
    for step in range(200):
        # Manually force curriculum switch
        rate_scale = 0.0 if step < 50 else 1.0
        
        optimizer.zero_grad()
        
        recon_logits, gates, gates_soft, z_qh, flat_idx, log_probs = model(
            input_ids, attention_mask=attention_mask, labels=labels, rate_scale=rate_scale
        )
        
        # Calculate active gates before we detach anything, just for logging
        active_gates = (gates > 0.5).float().sum(dim=1).mean().item()
        
        criterion_out = criterion(
            recon_logits=recon_logits,
            gates=gates,
            gates_soft=gates_soft,
            labels=labels,
            input_ids=input_ids,
            input_attention_mask=attention_mask,
            flat_idx=flat_idx,
            log_probs=log_probs,
            rate_scale=rate_scale,
            return_nli_per_item=False
        )
        
        total, distortion, nli, code_rate_loss, gate_rate_loss, sem_loss = criterion_out
        
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % 10 == 0 or step == 49 or step == 50 or step == 199:
            # Decode the first sequence to see what it's saying
            recon_ids = recon_logits.argmax(dim=-1)[0]
            # Filter out -100 from target comparison if needed, but for decode just use raw ids
            # we need to shift back or just decode the logits directly. The logits predict the *next* token.
            # To get a readable string, we just decode the argmax.
            recon_text = tokenizer.decode(recon_ids, skip_special_tokens=True)
            # truncate for display
            recon_text = recon_text[:50] + "..." if len(recon_text) > 50 else recon_text
            
            print(f"{step:>4} | {total.item():>6.2f} | {distortion.item():>6.2f} | {nli.item():>6.2f} | {gate_rate_loss.item():>6.2f} | {code_rate_loss.item():>6.2f} | {active_gates:>5.1f} | {recon_text}")

if __name__ == "__main__":
    test_overfit()
