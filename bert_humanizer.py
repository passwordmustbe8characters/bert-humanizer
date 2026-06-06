# bert_humanizer.py

import runpod
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np
import re

# --- 1. Load Model Once at Startup (Important for Performance!) ---
print("Loading DistilBERT model...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
model.eval()
print("Model loaded. Worker ready.")

# --- 2. Core Humanization Logic (Your working BERT approach) ---
def contextual_replace(text, mask_prob=0.15, top_k=10):
    """Replace ~15% of words with contextually appropriate alternatives using BERT."""
    words = text.split()
    if len(words) < 5:
        return text
    num_replace = max(1, int(len(words) * mask_prob))
    indices = np.random.choice(len(words), num_replace, replace=False)
    result_words = words.copy()

    for idx in indices:
        original = words[idx]
        masked_words = words.copy()
        masked_words[idx] = tokenizer.mask_token
        masked_text = " ".join(masked_words)

        inputs = tokenizer(masked_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)

        mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        if len(mask_token_index) == 0:
            continue

        logits = outputs.logits[0, mask_token_index, :]
        top_ids = torch.topk(logits, top_k).indices[0].tolist()

        replacement = original
        for tid in top_ids:
            cand = tokenizer.decode([tid]).strip()
            if cand and cand != original and cand[0].isalpha():
                replacement = cand
                break
        result_words[idx] = replacement

    return " ".join(result_words)

def cleanup_text(text):
    """Fix spacing and punctuation."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([,.!?])', r'\1', text)
    return text.strip()

# --- 3. RunPod Serverless Handler ---
def handler(event):
    """The main entry point for your serverless endpoint."""
    input_text = event.get("input", {}).get("text", "")
    if not input_text or len(input_text) < 20:
        return {"success": True, "text": input_text}

    # Apply BERT-based humanization
    augmented = contextual_replace(input_text, mask_prob=0.15, top_k=10)
    final = cleanup_text(augmented)

    return {"success": True, "text": final}

# --- 4. RunPod Serverless Starter (The missing piece that was causing the error) ---
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
