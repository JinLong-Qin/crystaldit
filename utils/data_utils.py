import torch
from tokenizers.normalizers import BertNormalizer
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import AutoModel

"""**VoCab Mapping and Normalizer**"""
f = open('vocab_mappings.txt', 'r')
mappings = f.read().strip().split('\n')
f.close()
mappings = {m[0]: m[2:] for m in mappings}
norm = BertNormalizer(lowercase=False, strip_accents=True, clean_text=True, handle_chinese_chars=True)

def normalize(text):
    text = [norm.normalize_str(s) for s in text.split('\n')]
    out = []
    for s in text:
        norm_s = ''
        for c in s:
            norm_s += mappings.get(c, ' ')
        out.append(norm_s)
    return '\n'.join(out)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# local_path = '/home/fall/.cache/huggingface/hub/models--m3rg-iitd--matscibert/snapshots/ced9d8f5f208712c4a90f98a246fe32155b29995'
local_path = '/home/fall/.cache/huggingface/hub/models--lfoppiano--MatTPUSciBERT/snapshots/be9727390b8032fd4c5108dda0b4637f3b31cece'

tokenizer = AutoTokenizer.from_pretrained(local_path, model_max_length=512)
text_model = AutoModel.from_pretrained(local_path)

# tokenizer = AutoTokenizer.from_pretrained(local_path, model_max_length=512)
# text_model = AutoModelForMaskedLM.from_pretrained(local_path)

print("Tokenizer vocab size:", len(tokenizer))
print("Model loaded OK:", text_model.__class__.__name__)

text_model.to(device)




def get_text_emb(text):
    norm_sents = normalize(text)
    encodings = tokenizer(norm_sents, return_tensors='pt', padding=True, truncation=True)
    if torch.cuda.is_available():
        encodings.to(device)
    with torch.no_grad():
        last_hidden_state = text_model(**encodings)[0]
    cls_emb = last_hidden_state[:, 0, :].cpu()
    return cls_emb