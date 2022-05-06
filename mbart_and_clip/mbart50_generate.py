from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import tqdm
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Device", device)
BATCH_SIZE = 16


with open("generated_english_captions.txt", "r") as fp:
    eng_captions = fp.readlines()
    eng_captions = [i.lstrip().rstrip() for i in eng_captions]


model = MBartForConditionalGeneration.from_pretrained("./results_mbart").to(device)
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer.src_lang = "en_XX"

aze_captions = []

# translate English to Aze
for i in tqdm.tqdm(range(0, len(eng_captions), BATCH_SIZE)):
    encoded_eng = tokenizer(eng_captions[i:i+BATCH_SIZE], return_tensors="pt", padding=True, truncation=True).to(device)
    generated_tokens = model.generate(**encoded_eng, forced_bos_token_id=tokenizer.lang_code_to_id["az_AZ"])
    aze_captions.extend(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))

# import pdb; pdb.set_trace()

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(device)

rus_captions = []

# translate English to Russian
for i in tqdm.tqdm(range(0, len(eng_captions), BATCH_SIZE)):
    encoded_eng = tokenizer(eng_captions[i:i+BATCH_SIZE], return_tensors="pt", padding=True, truncation=True).to(device)
    generated_tokens = model.generate(**encoded_eng, forced_bos_token_id=tokenizer.lang_code_to_id["ru_RU"])
    rus_captions.extend(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))


with open("generated_aze_captions.txt", "w") as fp:
    fp.write("\n".join(aze_captions))

with open("generated_rus_captions.txt", "w") as fp:
    fp.write("\n".join(rus_captions))
