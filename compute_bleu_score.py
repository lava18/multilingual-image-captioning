from nltk.translate.bleu_score import corpus_bleu
import json

f = open('sujay_translated_captions_rus.json')
# f = open('ameya_mBART_ru_preds.json')
# f = open('ru20k_predicted_captions_new.json')
data = json.load(f)

all_references, all_hypotheses = [], []
for item in data:
    refs, hyp = item['references'], item['hypothesis']
    
    temp = []
    for ref in refs:
        ref_tokens = ref.split(' ')
        temp.append(ref_tokens)
    all_references.append(temp)

    hyp_tokens = hyp.split(' ')
    all_hypotheses.append(hyp_tokens)

print(all_references[:2], all_hypotheses[:2])
bleu4 = corpus_bleu(all_references, all_hypotheses)
print("\nBLEU-4 score @ beam size of 1 is %.4f." % (bleu4))
