### Dataset

- Download the [Training (13GB)](http://images.cocodataset.org/zips/train2014.zip) and [Validation (6GB)](http://images.cocodataset.org/zips/val2014.zip) images. 
- Unzip `train2014.zip` and `val2014.zip` and move the image folders inside `caption_data`  

We will use [Andrej Karpathy's training, validation, and test splits](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip). This zip file contain the captions. 
- Move the `dataset_coco.json` file inside `caption_data`.

- Download [Russian](https://drive.google.com/file/d/1b-So963ud9taEbuQl7LDxf6u_pb8Dzpd/view?usp=sharing) and [Azerbaijani]() MSCOCO translated json files (based on MarianMT translations from en->ru and en->az).
(20k train, 5k val and 1k test examples)

# Training

Before you begin, make sure to save the required data files for training, validation, and testing. To do this, run the contents of [`create_input_files.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/create_input_files.py) after pointing it to the the Karpathy JSON file and the image folder `caption_data` containing the extracted `train2014` and `val2014` folders.

```
mkdir ru_outputs
python create_input_files.py
```

See [`train.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/train.py).

The parameters for the model (and training it) are at the beginning of the file, so you can easily check or modify them should you wish to.

To **train your model from scratch**, simply run this file –

`python train.py`

To **resume training at a checkpoint**, point to the corresponding file with the `checkpoint` parameter at the beginning of the code.

Note that we perform validation at the end of every training epoch.

## Adapter Fine-tuning

Notebooks for adapter fine-tuning for mBART on Russian and Marathi are present at: `/notebooks/Mr_MASSIVE_AdapterFinetuningMT.ipynb` `/notebooks/Ru_MASSIVE_AdapterFinetuningMT.ipynb`

# Inference and eval

See [`eval.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/eval.py), which implements this process for calculating the BLEU score on the validation set, with or without Beam Search.

### Model Checkpoint

You can download this pretrained model and the corresponding `word_map` [here](https://drive.google.com/open?id=189VY65I_n4RTpQnmLGj7IzVnOF6dmePC).

Note that this checkpoint should be [loaded directly with PyTorch](https://pytorch.org/docs/stable/torch.html?#torch.load), or passed to [`caption.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/caption.py) – see below.


__How do I compute all BLEU (i.e. BLEU-1 to BLEU-4) scores during evaluation?__

You'd need to modify the code in [`eval.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/eval.py) to do this. Please see [this excellent answer](<https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/issues/37#issuecomment-455924998>) by [kmario23](<https://github.com/kmario23>) for a clear and detailed explanation.
