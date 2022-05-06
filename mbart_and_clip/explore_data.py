import json


with open("Cross-lingual-Test-Dataset-XTD10/XTD10/test_image_names.txt", "r") as fp:
    adobe_ids = fp.readlines()
    adobe_ids = [i.lstrip().rstrip() for i in adobe_ids]

with open("annotations/instances_train2014.json", "r") as fp:
    train_data = json.load(fp)

with open("annotations/instances_val2014.json", "r") as fp:
    val_data = json.load(fp)

train_ids = [i["file_name"] for i in train_data["images"]]
val_ids = [i["file_name"] for i in val_data["images"]]

import pdb; pdb.set_trace()
