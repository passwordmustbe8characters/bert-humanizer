import runpod
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np
import re

# Load model once when the worker starts
print("Loading DistilBERT...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
model.eval()
print("Ready.")

def contextual_replace(text, mask_prob=0.15, top_k=10):
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
        mask_idx = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        if len(mask_idx) == 0:
            continue
        logits = outputs.logits[0, mask_idx, :]
        top_ids = torch.topk(logits, top_k).indices[0].tolist()
        replacement = original
        for tid in top_ids:
            cand = tokenizer.decode([tid]).strip()
            if cand and cand != original and cand[0].isalpha():
                replacement = cand
                break
        result_words[idx] = replacement
    return " ".join(result_words)

def cleanup(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([,.!?])', r'\1', text)
    return text.strip()

def handler(event):
    """Handler function for RunPod Serverless."""
    text = event.get("input", {}).get("text", "")
    if not text or len(text) < 20:
        return {"success": True, "text": text}
    augmented = contextual_replace(text, mask_prob=0.15, top_k=10)
    final = cleanup(augmented)
    return {"success": True, "text": final}

# REQUIRED entry point for RunPod Serverless
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
