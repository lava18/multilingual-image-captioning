from transformers import MarianMTModel, MarianTokenizer
import string

model_name = "Helsinki-NLP/opus-mt-en-ru"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to('cuda:0')

# src_text = [
#     "this is a sentence in english that we want to translate to azerbaijani",
#     "Canada is located in North America.",
#     "I have bad news for you."
# ]

preprocess_caption = lambda x: x.strip().translate(str.maketrans('', '', string.punctuation))

import json
import torch
from tqdm import tqdm
# Opening JSON file
f = open('caption_data/dataset_coco.json')
data = json.load(f)
src_text = []
batch_size = 256

ans = []
start_idx = 0
end_idx = 0
batch_no = 0
train_count = 0
val_count = 0
test_count = 0

ru_data = { "images": [] }

for img_id, img_dict in enumerate(tqdm(data['images'])):

  print(train_count, val_count, test_count)

  if train_count >= 20000 and val_count >= 5000 and test_count >= 1000:
    break

  if data['images'][img_id]['split'] in ("train", "restval"):
    if train_count >= 20000:
      continue
    else:
      train_count += 5
  elif data['images'][img_id]['split'] == "val":
    if val_count >= 5000:
      continue
    else:
      val_count += 5
  elif data['images'][img_id]['split'] == "test":
    if test_count >= 1000:
      continue
    else:
      test_count += 5

  for sent_id, sent_dict in enumerate(img_dict['sentences']):
    raw_sentence = sent_dict['raw']
    tokens = sent_dict['tokens']

    translated = model.generate(**tokenizer([raw_sentence], return_tensors="pt", padding=False).to('cuda:0'))
    tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    data['images'][img_id]['sentences'][sent_id].update({
      "raw": tgt_text,
      "tokens": preprocess_caption(tgt_text[0]).split(' ')
    })
    # if data['images'][img_id]['split'] == 'restval':
    #   data['images'][img_id]['split'] = 'val'

  ru_data["images"].append(data['images'][img_id])

with open("caption_data/ru_dataset_coco.json", "w", encoding='utf-8') as stream:
  stream.write(json.dumps(ru_data, ensure_ascii=False).encode('utf-8').decode())

# while end_idx < len(data):
#   end_idx = min(start_idx + batch_size, len(data))

#   items = data[start_idx: end_idx]
#   image_ids = [item['image_id'] for item in items]
#   src_text = [item['caption'] for item in items]
#   ids = [item['id'] for item in items]


# #   src_text = torch.Tensor([src_text]).to('cuda:0')
#   translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True).to('cuda:0'))
#   tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
# #   print(tgt_text)

#   for image_id, id, caption in zip(image_ids, ids, tgt_text):
#     ans.append({"image_id": image_id, "id": id, "caption": caption})

#   start_idx += batch_size
#   if end_idx >= len(data):
#     break

#   del translated
#   del tgt_text
#   del src_text
#   del items
#   del image_ids
#   del ids
#   torch.cuda.empty_cache()

#   batch_no += 1
#   print('Processed batch = ', batch_no)

# with open('russian_coco_captions.json', 'w') as f:
#   for item in ans:
#       f.write("%s," % item)