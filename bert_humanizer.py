import runpod
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import numpy as np
import re
import random

print("Loading DistilBERT model...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")
model.eval()
print("Model loaded. Ready for requests.")

def contextual_replace(text, mask_prob=0.15, top_k=10):
    words = text.split()
    if len(words) < 5:
        return text
    num_replace = max(1, int(len(words) * mask_prob))
    indices = random.sample(range(len(words)), num_replace)
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

def cleanup(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([,.!?])', r'\1', text)
    return text.strip()

@runpod.handler()
def handler(event):
    text = event.get("input", {}).get("text", "")
    if not text or len(text) < 20:
        return {"success": True, "text": text}
    augmented = contextual_replace(text, mask_prob=0.15, top_k=10)
    final = cleanup(augmented)
    return {"success": True, "text": final}