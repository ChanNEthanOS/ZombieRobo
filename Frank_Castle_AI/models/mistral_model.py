from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Using Mistral-7B for now. If you want a lighter model for real-time performance,
# consider alternatives like "EleutherAI/gpt-neo-1.3B" or "distilGPT2".
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"  # Let me know if you want to switch, bro.

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def generate_decision(prompt):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
