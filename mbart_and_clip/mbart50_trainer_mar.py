from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, TrainingArguments, Trainer
import json
import torch
from torch.nn.utils.rnn import pad_sequence

train_lang_file = "mar_eng/ted-train.orig.mar-eng"
val_lang_file = "mar_eng/ted-dev.orig.mar-eng"

source_utt_train = []
target_utt_train = []
with open(train_lang_file, "r") as fp:
    lines = [i.strip() for i in fp.readlines()]
    for line in lines:
        source = line.split("|||")[1]
        target = line.split("|||")[0]
        source_utt_train.append(source)
        target_utt_train.append(target)


source_utt_val = []
target_utt_val = []
with open(val_lang_file, "r") as fp:
    lines = [i.strip() for i in fp.readlines()]
    for line in lines:
        source = line.split("|||")[1]
        target = line.split("|||")[0]
        source_utt_val.append(source)
        target_utt_val.append(target)

print("Training samples", len(source_utt_train))
print("Valid samples", len(source_utt_val))

assert len(source_utt_train) == len(target_utt_train)
assert len(source_utt_val) == len(target_utt_val)

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

class Eng2MarDataset(torch.utils.data.Dataset):

    def __init__(self, eng_utterances, mar_utterances, feature_extractor):
        self.eng = eng_utterances
        self.mar = mar_utterances
        assert len(self.eng) == len(self.mar)
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.eng)

    def __getitem__(self, idx):

        encoding = self.feature_extractor(self.eng[idx], return_tensors="pt")
        with self.feature_extractor.as_target_tokenizer():
            labels = self.feature_extractor(self.mar[idx], return_tensors="pt").input_ids

        encoding["labels"] = labels
        #import pdb; pdb.set_trace()
        return encoding#, labels


model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
feature_extractor = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="mr_IN", padding=True, truncation=True)

train_dataset = Eng2MarDataset(source_utt_train, target_utt_train, feature_extractor)
val_dataset = Eng2MarDataset(source_utt_val, target_utt_val, feature_extractor)
#test_dataset = Eng2MarDataset(source_utt_test, target_utt_test, feature_extractor)

# import pdb; pdb.set_trace()

training_args = TrainingArguments(
    output_dir="./results_mbart_mar",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=1,
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

