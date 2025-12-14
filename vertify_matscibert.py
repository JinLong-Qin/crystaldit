from transformers import AutoTokenizer, AutoModelForMaskedLM

local_path = "/share/users/rgzndrone/.cache/huggingface/hub/models--m3rg-iitd--matscibert/snapshots/ced9d8f5f208712c4a90f98a246fe32155b29995"

tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoModelForMaskedLM.from_pretrained(local_path)

print("Tokenizer vocab size:", len(tokenizer))
print("Model loaded OK:", model.__class__.__name__)
