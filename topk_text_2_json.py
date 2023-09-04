from transformers import AutoTokenizer
from argparse import ArgumentParser
import json
import os

parser = ArgumentParser()

parser.add_argument('--file', required=True)
parser.add_argument('--save_to', required=True)
parser.add_argument('--tokenizer', default='bert-base-uncased')
parser.add_argument('--truncate', type=int, default=512)
parser.add_argument('--q_truncate', type=int, default=-1)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
SEP = tokenizer.sep_token

def encode_pass_item(item):
    qid, did, qry, body = item.strip().split('\t')
    qry_encoded = tokenizer.encode(
        qry,
        truncation=True if args.q_truncate else False,
        max_length=args.q_truncate,
        add_special_tokens=False,
        padding=False,
    )
    doc_encoded = tokenizer.encode(
        body,
        truncation=True,
        max_length=args.truncate,
        add_special_tokens=False,
        padding=False
        )

    if len(doc_encoded) == 0:
        raise ValueError(f'Please check the text of {did} for query {qid}!')

    entry = {
        'qid': qid,
        'pid': did,
        'qry': qry_encoded,
        'psg': doc_encoded,
    }
    entry = json.dumps(entry)

    return entry


if os.path.exists(args.save_to):
    raise FileExistsError(f'Save file {args.save_to} already exists')

with open(args.save_to, 'w') as jfile:
    if args.q_truncate < 0:
        print('queries are not truncated', flush=True)
        args.q_truncate = None

    with open(args.file) as f:
        for line in f:
            json_item = encode_pass_item(line)
            jfile.write(json_item + '\n')