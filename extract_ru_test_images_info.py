import json
from tqdm import tqdm
# Opening JSON file
f = open('caption_data/ru_dataset_coco.json')
data = json.load(f)


saved_data = []
for img_id, img_dict in enumerate(tqdm(data['images'])):
    split = img_dict['split']
    filename = img_dict['filename']
    filepath = img_dict['filepath']

    if split == 'test':
        refs = []
        for sent_id, sent_dict in enumerate(img_dict['sentences']):
            extracted_image_id = sent_dict['imgid']
            ref = sent_dict['raw'][0]
            refs.append(ref)
    
        saved_data.append({'image_id': extracted_image_id, 
                            'filepath': filepath, 
                            'filename': filename, 
                            'references': refs})

with open("caption_data/ru_test_images_dataset_coco.json", "w", encoding='utf-8') as stream:
  stream.write(json.dumps(saved_data, ensure_ascii=False).encode('utf-8').decode())





            