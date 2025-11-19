# eval_snippet.py (high-level)
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np

#MODEL = "meta-llama/Llama-3-8b"       # base
MODEL = "meta-llama/Llama-2-7b-hf"
FT_MODEL = "./llama2-edu-qlora/lora_adapter"  # fine-tuned adapters

tokenizer = AutoTokenizer.from_pretrained(MODEL)
# Load base & fine-tuned (load base and apply adapters for ft)
base = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto")
# for ft: same base but load peft adapters
from peft import PeftModel
ft = PeftModel.from_pretrained(base, FT_MODEL)

def generate_answer(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=128, do_sample=False, temperature=0.0)
    return tokenizer.decode(out[0], skip_special_tokens=True)

# For each dataset item: build same prompt used in training & parse final answer
# compute accuracy by string/numeric matching or choice matching
prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
answer_base = generate_answer(base, prompt)
answer_ft = generate_answer(ft, prompt)
print(answer_base)
print(answer_ft)
