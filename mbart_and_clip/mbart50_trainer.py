from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, TrainingArguments, Trainer
import json
import torch
from torch.nn.utils.rnn import pad_sequence

source_lang_file = "1.0/data/en-US.jsonl"
target_lang_file = "1.0/data/az-AZ.jsonl"

with open(source_lang_file, "r") as fp:
    source_lang_data = [json.loads(line) for line in fp]

with open(target_lang_file, "r") as fp:
    target_lang_data = [json.loads(line) for line in fp]

assert len(source_lang_data) == len(target_lang_data)

source_id_to_utterance_train = {i["id"]: i["utt"] for i in source_lang_data if i["partition"] == "train"}
target_id_to_utterance_train = {i["id"]: i["utt"] for i in target_lang_data if i["partition"] == "train"}

source_id_to_utterance_val = {i["id"]: i["utt"] for i in source_lang_data if i["partition"] == "dev"}
target_id_to_utterance_val = {i["id"]: i["utt"] for i in target_lang_data if i["partition"] == "dev"}

source_id_to_utterance_test = {i["id"]: i["utt"] for i in source_lang_data if i["partition"] == "test"}
target_id_to_utterance_test = {i["id"]: i["utt"] for i in target_lang_data if i["partition"] == "test"}

source_utt_train = []
target_utt_train = []
for idx, utt in source_id_to_utterance_train.items():
    source_utt_train.append(utt)
    target_utt_train.append(target_id_to_utterance_train[idx])

source_utt_val = []
target_utt_val = []
for idx, utt in source_id_to_utterance_val.items():
    source_utt_val.append(utt)
    target_utt_val.append(target_id_to_utterance_val[idx])

source_utt_test = []
target_utt_test = []
for idx, utt in source_id_to_utterance_test.items():
    source_utt_test.append(utt)
    target_utt_test.append(target_id_to_utterance_test[idx])


print("Training samples", len(source_utt_train))
print("Valid samples", len(source_utt_val))
print("Test samples", len(source_utt_test))

assert len(source_utt_train) == len(target_utt_train)
assert len(source_utt_val) == len(target_utt_val)
assert len(source_utt_test) == len(target_utt_test)

def collate_fn(batch):
    all_input_ids = []
    all_att_masks = []
    all_labels = []
    for x in batch:
        all_input_ids.append(x["input_ids"].squeeze(0))
        all_att_masks.append(x["attention_mask"].squeeze(0))
        all_labels.append(x["labels"].squeeze(0))

    all_input_ids_pad = pad_sequence(all_input_ids, batch_first=True)
    all_att_masks_pad = pad_sequence(all_att_masks, batch_first=True)
    all_labels_pad = pad_sequence(all_labels, batch_first=True)

    return {
        'input_ids': all_input_ids_pad,
        'attention_mask': all_att_masks_pad,
        'labels': all_labels_pad
    }

class Eng2AzeDataset(torch.utils.data.Dataset):

    def __init__(self, eng_utterances, aze_utterances, feature_extractor):
        self.eng = eng_utterances
        self.aze = aze_utterances
        assert len(self.eng) == len(self.aze)
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.eng)

    def __getitem__(self, idx):

        encoding = self.feature_extractor(self.eng[idx], return_tensors="pt")
        with self.feature_extractor.as_target_tokenizer():
            labels = self.feature_extractor(self.aze[idx], return_tensors="pt").input_ids

        encoding["labels"] = labels
        #import pdb; pdb.set_trace()
        return encoding#, labels


model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
feature_extractor = MBart50TokenizerFast.from_pretrained("mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="az_AZ", padding=True, truncation=True)

train_dataset = Eng2AzeDataset(source_utt_train, target_utt_train, feature_extractor)
val_dataset = Eng2AzeDataset(source_utt_val, target_utt_val, feature_extractor)
test_dataset = Eng2AzeDataset(source_utt_test, target_utt_test, feature_extractor)

# import pdb; pdb.set_trace()

training_args = TrainingArguments(
    output_dir="./results_mbart",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    tokenizer=feature_extractor
)

trainer.train()
trainer.save_model()

