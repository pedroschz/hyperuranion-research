import torch
from sentence_transformers import SentenceTransformer
from model import SemanticAutoencoder
from transformers import BartTokenizer

def test_old_architecture(text, model_path="seq2seq_simplification_epoch_1.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    
    # Initialize the Model
    model = SemanticAutoencoder().to(device)
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    
    sbert = SentenceTransformer("sentence-transformers/all-distilroberta-v1").to(device)
    
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        # Use standard generate from the base BART model
        # The model was trained with standard Seq2Seq loss but heavily penalized for length
        base_model = model.bart.get_base_model()
        
        generated_ids = base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=64,
            num_beams=4,
            early_stopping=True
        )
        
        reconstructed_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        print("\n--- Original Text ---")
        print(text)
        print("\n--- Model Output (Attempted Compression) ---")
        print(reconstructed_text)
        
        orig_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        orig_bits = orig_tokens * 16 
        
        gen_tokens = len(generated_ids[0]) - 2
        if gen_tokens < 0: gen_tokens = 0
        simp_bits = gen_tokens * 16
        
        compression_rate = (1 - simp_bits / orig_bits) * 100 if orig_bits > 0 else 0
        
        print(f"\n--- Output Compression Stats ---")
        print(f"Original: {orig_tokens} tokens -> {orig_bits} bits")
        print(f"Output: {gen_tokens} tokens -> {simp_bits} bits")
        print(f"Absolute Storage Savings: {orig_bits - simp_bits} bits")
        print(f"Bit-level Compression Rate: {compression_rate:.2f}%")
        
        orig_emb = sbert.encode(text, convert_to_tensor=True)
        rec_emb = sbert.encode(reconstructed_text, convert_to_tensor=True)
        cos_sim = torch.nn.functional.cosine_similarity(orig_emb.unsqueeze(0), rec_emb.unsqueeze(0))
        
        print("\n--- Semantic Fidelity ---")
        print(f"Cosine Similarity Score: {cos_sim.item():.4f}")

if __name__ == "__main__":
    test_string = "The rapid proliferation of artificial intelligence technologies has simultaneously catalyzed unprecedented efficiencies across multiple industrial sectors and precipitated profound anxieties regarding the potential displacement of human labor."
    test_old_architecture(test_string)