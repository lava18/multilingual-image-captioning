from comet import download_model, load_from_checkpoint
import json

with open("marathi_cometqe.json", "r") as fp:
    data = json.load(fp)

model_path = download_model("wmt21-comet-qe-mqm")
model = load_from_checkpoint(model_path)

seg_scores, sys_score = model.predict(data, batch_size=8, gpus=1)

print("Comet QE score", sys_score)
