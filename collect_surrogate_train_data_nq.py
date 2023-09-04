import json
import random

def clean_text(text):
    char = []
    for x in text:
        if x.isprintable():
            char.append(x)
        else:
            char.append(' ')
    c_text = ''.join(char)
    return c_text


nq_train_file = './dpr_nq/biencoder-nq-train.json'
with open(nq_train_file,'r') as load_f:
    load_dict = json.load(load_f)

neg_num = 8
output_train_triples = f'./dpr_nq/train_data/biencoder-nq-train-triples-p1n{neg_num}.tsv'

with open(output_train_triples, 'w') as w:
    for i, json_sample in enumerate(load_dict):
        query_id = json_sample['dataset'] + '-{}'.format(i)
        query_text = json_sample['question']

        positive_ctxs = json_sample['positive_ctxs']
        negative_ctxs = json_sample['negative_ctxs']

        pos_passages = {}
        for ctx in positive_ctxs:
            if ctx['passage_id'] not in pos_passages:
                pos_passages[ctx['passage_id']] = ctx['text']

        neg_passages = {}
        for ctx in negative_ctxs:
            if ctx['passage_id'] not in neg_passages:
                neg_passages[ctx['passage_id']] = ctx['text']
        neg_pass_ids = list(neg_passages.keys())

        for pos_id in pos_passages.keys():
            pos_text = clean_text(pos_passages[pos_id])

            if len(neg_pass_ids) >= neg_num:
                sampled_neg_ids = random.sample(neg_pass_ids, k=neg_num)
            else:
                sampled_neg_ids = random.sample(neg_pass_ids, k=len(neg_pass_ids))
                while len(sampled_neg_ids) < neg_num:
                    sampled_neg_ids.extend(random.sample(neg_pass_ids, k=1))

            assert len(sampled_neg_ids) == neg_num

            random.shuffle(sampled_neg_ids)
            for neg_id in sampled_neg_ids:
                neg_text = clean_text(neg_passages[neg_id])
                int(pos_id)
                int(neg_id)
                w.write('\t'.join([query_id, pos_id, neg_id, query_text, pos_text, neg_text]) + '\n')