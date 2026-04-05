import torch
from datasets import load_dataset
from transformers import BartTokenizer
from collections import Counter
from tqdm import tqdm

def main():
    print("Loading dataset...")
    dataset = load_dataset("eilamc14/wikilarge-clean", split="train")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    
    # Sample subset to count frequencies quickly
    subset = dataset.select(range(min(50000, len(dataset))))
    
    counter = Counter()
    print("Counting token frequencies in 'target' text to find semantic primitives...")
    for example in tqdm(subset):
        tokens = tokenizer.encode(example["target"], add_special_tokens=False)
        counter.update(tokens)
        
    # We want exactly 2048 tokens total (11 bits)
    VOCAB_LIMIT = 2048
    
    # First, always include special tokens to ensure generation works properly
    special_ids = set(tokenizer.all_special_ids)
    
    allowed_ids = list(special_ids)
    remaining_slots = VOCAB_LIMIT - len(allowed_ids)
    
    # Get most common tokens
    most_common = counter.most_common()
    for token_id, freq in most_common:
        if remaining_slots <= 0:
            break
        if token_id not in allowed_ids:
            allowed_ids.append(token_id)
            remaining_slots -= 1
            
    # If we still have slots (unlikely), pad with random valid ids
    for i in range(tokenizer.vocab_size):
        if remaining_slots <= 0:
            break
        if i not in allowed_ids:
            allowed_ids.append(i)
            remaining_slots -= 1
                
    allowed_ids_tensor = torch.tensor(allowed_ids, dtype=torch.long)
    torch.save(allowed_ids_tensor, "allowed_vocab.pt")
    
    print(f"Saved {len(allowed_ids_tensor)} allowed tokens to allowed_vocab.pt")

if __name__ == "__main__":
    main()