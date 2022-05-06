import os
from PIL import Image
import requests
import json
from tqdm import tqdm
from io import BytesIO

IMAGES_BASE_PATH = "/home/ec2-user/images"
if not os.path.exists(IMAGES_BASE_PATH):
    os.makedirs(IMAGES_BASE_PATH)

IMAGES_TO_EXCLUDE = set()

def get_image_from_url(url, image_id):
    img_save_path = os.path.join(IMAGES_BASE_PATH, f"{image_id}")
    if os.path.exists(img_save_path):
        return
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img.save(f"{img_save_path}")
    except:
        IMAGES_TO_EXCLUDE.add(image_id)


with open("Cross-lingual-Test-Dataset-XTD10/XTD10/test_image_names.txt", "r") as fp:
    adobe_ids = fp.readlines()
    adobe_ids = [i.lstrip().rstrip() for i in adobe_ids]

# wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip and unzip
with open("annotations/instances_train2014.json", "r") as fp:
    train_data = json.load(fp)

with open("annotations/instances_val2014.json", "r") as fp:
    val_data = json.load(fp)

train_ids = {i["file_name"]: i["coco_url"] for i in train_data["images"]}
val_ids = {i["file_name"]: i["coco_url"] for i in val_data["images"]}

train_ids.update(val_ids)

all_ids_set = set(list(train_ids.keys()))
assert len(all_ids_set.intersection(set(adobe_ids))) == 1000

for i in tqdm(adobe_ids):
    get_image_from_url(train_ids[i], i)
