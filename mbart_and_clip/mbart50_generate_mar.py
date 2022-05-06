from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import tqdm
import torch
import json

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Device", device)
BATCH_SIZE = 16


with open("eng_predicted_captions_new.json", "r") as fp:
    eng_captions_data = json.load(fp)
    eng_captions = [i["hypothesis"].lstrip().rstrip() for i in eng_captions_data]

with open("ru20k_predicted_captions_new.json", "r") as fp:
    rus_captions_data = json.load(fp)

assert len(eng_captions_data) == len(rus_captions_data)

model = MBartForConditionalGeneration.from_pretrained("./results_mbart_mar").to(device)
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer.src_lang = "en_XX"

mar_captions = []

# translate English to Mar
for i in tqdm.tqdm(range(0, len(eng_captions), BATCH_SIZE)):
    encoded_eng = tokenizer(eng_captions[i:i+BATCH_SIZE], return_tensors="pt", padding=True, truncation=True).to(device)
    generated_tokens = model.generate(**encoded_eng, forced_bos_token_id=tokenizer.lang_code_to_id["mr_IN"])
    mar_captions.extend(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))

# import pdb; pdb.set_trace()

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(device)

rus_captions = []

# translate English to Russian
for i in tqdm.tqdm(range(0, len(eng_captions), BATCH_SIZE)):
    encoded_eng = tokenizer(eng_captions[i:i+BATCH_SIZE], return_tensors="pt", padding=True, truncation=True).to(device)
    generated_tokens = model.generate(**encoded_eng, forced_bos_token_id=tokenizer.lang_code_to_id["ru_RU"])
    rus_captions.extend(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))

assert len(eng_captions_data) == len(mar_captions) == len(rus_captions) == len(rus_captions_data)

data_cometqe_mar = []
data_cometqe_rus = []
for i in range(len(eng_captions_data)):
    eng_captions_data[i]["mar_hyp"] = mar_captions[i]
    rus_captions_data[i]["hypothesis"] = rus_captions[i]
    data_cometqe_mar.append({
        "src": eng_captions_data[i]["hypothesis"],
        "mt": mar_captions[i]
    })
    data_cometqe_rus.append({
        "src": eng_captions_data[i]["hypothesis"],
        "mt": rus_captions[i]
    })

with open("translated_captions_mar.json", "w") as fp:
    json.dump(eng_captions_data, fp)

with open("translated_captions_rus.json", "w") as fp:
    json.dump(rus_captions_data, fp)

with open("marathi_cometqe.json", "w") as fp:
    json.dump(data_cometqe_mar, fp)

with open("russian_cometqe.json", "w") as fp:
    json.dump(data_cometqe_rus, fp)
