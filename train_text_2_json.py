from transformers import AutoTokenizer
from argparse import ArgumentParser
import json

parser = ArgumentParser()

parser.add_argument('--train_file', required=True)
parser.add_argument('--save_to_file', required=True)
parser.add_argument('--tokenizer', default='bert-base-uncased')
parser.add_argument('--truncate', type=int, default=512)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
SEP = tokenizer.sep_token

def encode_item(item):
    qid, pos_did, neg_did, query, pos_body, neg_body =  item.strip().split('\t')

    qry_encoded = tokenizer.encode(
        query,
        truncation=True,
        max_length=args.truncate,
        add_special_tokens=False,
        padding=False,
    )
    query_dict = {
        'qid': qid,
        'query': qry_encoded,
    }

    pos_doc_encoded = tokenizer.encode(
            pos_body,
            truncation=True,
            max_length=args.truncate,
            add_special_tokens=False,
            padding=False
    )
    pos_dict = {
        'pid': pos_did,
        'passage': pos_doc_encoded,
    }

    neg_doc_encoded = tokenizer.encode(
        neg_body,
        truncation=True,
        max_length=args.truncate,
        add_special_tokens=False,
        padding=False
    )
    neg_dict = {
        'pid': neg_did,
        'passage': neg_doc_encoded,
    }

    hinge_item_set = {
        'qry': query_dict,
        'pos': pos_dict,
        'neg': neg_dict
    }

    hinge_entry = json.dumps(hinge_item_set)
    return hinge_entry


with open(args.save_to_file, 'w') as jfile:
    with open(args.train_file) as train_file:
        for line in train_file:
            hinge_json_item = encode_item(line)
            jfile.write(hinge_json_item + '\n')