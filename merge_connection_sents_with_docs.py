import os
import torch
from transformers import BertTokenizerFast, BertForNextSentencePrediction, BertForSequenceClassification, BertConfig
from tqdm import tqdm
from argparse import ArgumentParser
import spacy
nlp = spacy.load('en_core_web_sm')


class BertNSPScorer(object):
    def __init__(self, device):
        self.model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(device)
        self.model.eval()
        self.device = device

    def next_sentence_score(self, inputs):
        if isinstance(inputs, str):
            inputs = [inputs]

        inputs = self.tokenizer.batch_encode_plus(inputs, max_length=512, padding='longest',
                                                  truncation='longest_first', return_tensors="pt").to(self.device)

        outputs = self.model(**inputs)
        logits = outputs.logits
        nsp_score = logits[:, 0].detach().cpu().numpy().tolist()
        nsp_gap_score = (logits[:, 0] - logits[:, 1]).detach().cpu().numpy().tolist()
        softmax_nsp_score = torch.softmax(logits, -1)[:, 0].detach().cpu().numpy().tolist()
        return nsp_score, nsp_gap_score, softmax_nsp_score


class BertRelScorer(object):
    def __init__(self, model_dir, device, num_labels=1):
        self.num_labels = num_labels
        self.config = BertConfig.from_pretrained(model_dir, num_labels=self.num_labels)
        self.model = BertForSequenceClassification.from_pretrained(model_dir, config=self.config)
        for param in self.model.parameters():
            param.requires_grad = False
        if not os.path.exists(model_dir + '/vocab.txt'):
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained(model_dir)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def relevance(self, inputs):
        if isinstance(inputs, str):
            inputs = [inputs]

        inputs = self.tokenizer.batch_encode_plus(inputs, max_length=512, padding='longest',
                                                  truncation='only_second', return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        if self.num_labels == 1:
            rel_score = logits[:, 0].detach().cpu().numpy().tolist()
        elif self.num_labels == 2:
            rel_score = torch.softmax(logits, -1)[:, 1].detach().cpu().numpy().tolist()
        else:
            raise NotImplementedError
        return rel_score


def compute_read_coh_scores(connect_sents, output_nsp_file, collection, device, bs=32):
    if not os.path.exists(output_nsp_file):
        print('Computing NSP scores by pre-trained BERT model...')
        nsp_model = BertNSPScorer(device)
        queries = list(connect_sents.keys())
        with open(output_nsp_file, 'w') as w:
            for q_id in tqdm(queries):
                input_ids = []
                input_texts = []
                for p_id in connect_sents[q_id]:
                    p_body = collection[p_id]
                    p_sents = [str(s).strip() for s in nlp(p_body).sents]
                    for sent_id in connect_sents[q_id][p_id]:
                        sent_text = connect_sents[q_id][p_id][sent_id]
                        local_input_ids = []
                        local_input_texts = []
                        for idx in range(len(p_sents) + 1):
                            #### two parts
                            p_sents_insert = list(p_sents)
                            p_sents_insert.insert(idx, sent_text)
                            if idx == 0:
                                text_front = p_sents_insert[0]
                                text_behind = ' '.join(p_sents_insert[1:])
                                input_id = sent_id + '-{}'.format(idx)
                                local_input_ids.append(input_id)
                                local_input_texts.append([text_front, text_behind])
                            elif idx == len(p_sents):
                                text_front = ' '.join(p_sents_insert[:-1])
                                text_behind = p_sents_insert[-1]
                                input_id = sent_id + '-{}'.format(idx)
                                local_input_ids.append(input_id)
                                local_input_texts.append([text_front, text_behind])
                            else:
                                text_front1 = ' '.join(p_sents_insert[:idx])
                                text_behind1 = ' '.join(p_sents_insert[idx:])
                                input_id1 = sent_id + '-{}#{}'.format(idx, 0)
                                text_front2 = ' '.join(p_sents_insert[:idx+1])
                                text_behind2 = ' '.join(p_sents_insert[idx+1:])
                                input_id2 = sent_id + '-{}#{}'.format(idx, 1)
                                local_input_ids.append(input_id1)
                                local_input_ids.append(input_id2)
                                local_input_texts.append([text_front1, text_behind1])
                                local_input_texts.append([text_front2, text_behind2])

                        for input_id, input_text in zip(local_input_ids, local_input_texts):
                            input_ids.append(input_id)
                            input_texts.append(input_text)
                            if len(input_ids) == bs:
                                with torch.no_grad():
                                    nsp_list, gap_nsp_list, softmax_nsp_list = nsp_model.next_sentence_score(inputs=input_texts)
                                for i, input_id in enumerate(input_ids):
                                    w.write(f'{q_id}\t{input_id}\t{nsp_list[i]}\t{gap_nsp_list[i]}\t{softmax_nsp_list[i]}\n')
                                input_ids = []
                                input_texts = []

                if len(input_ids) > 0:
                    assert len(input_ids) < bs
                    with torch.no_grad():
                        nsp_list, gap_nsp_list, softmax_nsp_list = nsp_model.next_sentence_score(inputs=input_texts)
                    for i, input_id in enumerate(input_ids):
                        w.write(f'{q_id}\t{input_id}\t{nsp_list[i]}\t{gap_nsp_list[i]}\t{softmax_nsp_list[i]}\n')

    print(f'Loading NSP scores from {output_nsp_file} ...')
    pfp_nsp_scores = {}
    with open(output_nsp_file) as f:
        for line in f:
            q_id, pfp_id, _, pfp_gap_nsp, _ = line.strip().split() ### gap score is used.
            if q_id not in pfp_nsp_scores:
                pfp_nsp_scores[q_id] = {}
            if '#' in pfp_id:
                pfp_id = pfp_id.split('#')[0]
            if pfp_id not in pfp_nsp_scores[q_id]:
                pfp_nsp_scores[q_id][pfp_id] = []
            pfp_nsp_scores[q_id][pfp_id].append(float(pfp_gap_nsp))

    all_coh_scores = {}
    fp_coh_scores = {}
    for q_id in pfp_nsp_scores.keys():
        all_coh_scores[q_id] = {}
        fp_coh_scores[q_id] = {}
        for pfp_id in pfp_nsp_scores[q_id]:
            assert len(pfp_nsp_scores[q_id][pfp_id]) in [1, 2]
            coh = sum(pfp_nsp_scores[q_id][pfp_id]) / len(pfp_nsp_scores[q_id][pfp_id])
            fp_coh_scores[q_id][pfp_id] = coh

            p_id = pfp_id.split('-')[0]
            if p_id not in all_coh_scores[q_id]:
                all_coh_scores[q_id][p_id] = []
            all_coh_scores[q_id][p_id].append(coh)

    min_max_coh_scores = {}
    for q_id in all_coh_scores.keys():
        min_max_coh_scores[q_id] = {}
        for p_id in all_coh_scores[q_id]:
            min_max_coh_scores[q_id][p_id] = {'min': min(all_coh_scores[q_id][p_id]),
                                              'max': max(all_coh_scores[q_id][p_id])}

    norm_coh_scores = {}
    for q_id in fp_coh_scores.keys():
        if q_id not in norm_coh_scores:
            norm_coh_scores[q_id] = {}
        for fp_id in fp_coh_scores[q_id].keys():
            fp_coh = fp_coh_scores[q_id][fp_id]
            p_id = fp_id.split('-')[0]
            if p_id not in norm_coh_scores[q_id]:
                norm_coh_scores[q_id][p_id] = {}
            if len(all_coh_scores[q_id][p_id]) == 1:
                assert fp_coh == min_max_coh_scores[q_id][p_id]['max'] == min_max_coh_scores[q_id][p_id]['min']
                norm_coh_score = 1.
            else:
                norm_coh_score = (fp_coh - min_max_coh_scores[q_id][p_id]['min']) / \
                                 (min_max_coh_scores[q_id][p_id]['max'] - min_max_coh_scores[q_id][p_id]['min'])
            norm_coh_scores[q_id][p_id][fp_id] = norm_coh_score

    return norm_coh_scores


def compute_read_rel_scores(surrogate_model, connect_sents, output_rel_file, qry_collection, collection, device, bs=32):
    if not os.path.exists(output_rel_file):
        print('Computing Rel scores by surrogate model...')
        rel_model = BertRelScorer(surrogate_model, device, num_labels=1)
        queries = list(connect_sents.keys())
        with open(output_rel_file, 'w') as w:
            for q_id in tqdm(queries):
                input_ids = []
                input_texts = []
                for p_id in connect_sents[q_id]:
                    for sent_id in connect_sents[q_id][p_id]:
                        p_body = collection[p_id]
                        p_sents = [str(s).strip() for s in nlp(p_body).sents]
                        for idx in range(len(p_sents) + 1):
                            input_id = sent_id + '-{}'.format(idx)
                            input_ids.append(input_id)
                            p_sents_insert = list(p_sents)
                            p_sents_insert.insert(idx, connect_sents[q_id][p_id][sent_id])
                            p_text_insert = ' '.join(p_sents_insert)
                            input_texts.append([qry_collection[q_id], p_text_insert])
                            if len(input_ids) == bs:
                                with torch.no_grad():
                                    rel_list = rel_model.relevance(inputs=input_texts)
                                for i, input_id in enumerate(input_ids):
                                    w.write(f'{q_id}\t{input_id}\t{rel_list[i]}\n')
                                input_ids = []
                                input_texts = []

                if len(input_ids) > 0:
                    with torch.no_grad():
                        rel_list = rel_model.relevance(inputs=input_texts)
                    for i, input_id in enumerate(input_ids):
                        w.write(f'{q_id}\t{input_id}\t{rel_list[i]}\n')

    print(f'Loading Rel scores from {output_rel_file} ...')
    rel_scores = {}
    min_max_rel_scores = {}
    with open(output_rel_file) as f:
        for line in f:
            q_id, fp_id, rel, = line.strip().split()
            if q_id not in rel_scores:
                rel_scores[q_id] = {}
            p_id = fp_id.split('-')[0]
            if p_id not in rel_scores[q_id]:
                rel_scores[q_id][p_id] = []
            rel_scores[q_id][p_id].append(float(rel))

    for q_id in rel_scores.keys():
        min_max_rel_scores[q_id] = {}
        for p_id in rel_scores[q_id]:
            min_max_rel_scores[q_id][p_id] = {'min': min(rel_scores[q_id][p_id]),
                                              'max': max(rel_scores[q_id][p_id])}

    norm_rel_scores = {}
    with open(output_rel_file) as f:
        for line in f:
            q_id, fp_id, fp_rel, = line.strip().split()
            if q_id not in norm_rel_scores:
                norm_rel_scores[q_id] = {}
            p_id = fp_id.split('-')[0]
            if p_id not in norm_rel_scores[q_id]:
                norm_rel_scores[q_id][p_id] = {}
            if len(rel_scores[q_id][p_id]) == 1:
                assert float(fp_rel) == min_max_rel_scores[q_id][p_id]['max'] == min_max_rel_scores[q_id][p_id]['min']
                norm_rel_score = 1.
            else:
                norm_rel_score = (float(fp_rel) - min_max_rel_scores[q_id][p_id]['min']) / \
                                 (min_max_rel_scores[q_id][p_id]['max'] - min_max_rel_scores[q_id][p_id]['min'])
            norm_rel_scores[q_id][p_id][fp_id] = norm_rel_score

    return norm_rel_scores


def merging_connect_sents(connect_sents, collection, qry_collection, norm_coh_scores, norm_rel_scores, target_file,
                          output_qry_sent, output_qry_sent_body, coh_weight=0.5, rel_weight=0.5):

    weighted_norm_coh_rel_scores = {}
    for q_id in norm_coh_scores.keys():
        if q_id not in weighted_norm_coh_rel_scores:
            weighted_norm_coh_rel_scores[q_id] = {}
        for p_id in norm_coh_scores[q_id].keys():
            if p_id not in weighted_norm_coh_rel_scores[q_id]:
                weighted_norm_coh_rel_scores[q_id][p_id] = []
            for fp_id in norm_coh_scores[q_id][p_id].keys():
                norm_coh_rel_score = coh_weight * norm_coh_scores[q_id][p_id][fp_id] + rel_weight * norm_rel_scores[q_id][p_id][fp_id]
                weighted_norm_coh_rel_scores[q_id][p_id].append((norm_coh_rel_score, fp_id))

    top_connect_sents = {}
    for q_id in weighted_norm_coh_rel_scores.keys():
        if q_id not in top_connect_sents:
            top_connect_sents[q_id] = {}
        for p_id in weighted_norm_coh_rel_scores[q_id]:
            sorted_connect_sents = sorted(weighted_norm_coh_rel_scores[q_id][p_id], reverse=True)
            top_s, top_id = sorted_connect_sents[0]
            assert len(top_id.split('-')) == 3
            connect_id = '-'.join(top_id.split('-')[:-1])
            insert_id = int(top_id.split('-')[-1])
            top_connect_sent = connect_sents[q_id][p_id][connect_id]

            p_body = collection[p_id]
            p_sents = [str(s).strip() for s in nlp(p_body).sents]
            p_sents.insert(insert_id, top_connect_sent)
            p_text_insert = ' '.join(p_sents)

            if p_id not in top_connect_sents[q_id]:
                top_connect_sents[q_id][p_id] = (top_id, top_connect_sent, p_text_insert)
            else:
                raise KeyError

    with open(target_file) as f, \
        open(output_qry_sent, 'w') as w1, \
        open(output_qry_sent_body, 'w') as w2:
        for line in f:
            q_id, p_id, _, _ = line.strip().split()
            if q_id not in top_connect_sents:
                print(f'Note: fail to generate connection sentences to target doc {p_id} for query {q_id}.')
                w1.write(f'{q_id}\t{p_id}\t{qry_collection[q_id]}\t{collection[p_id]}\n')
                w2.write(f'{q_id}\t{p_id}\t{qry_collection[q_id]}\t"None"\n')
            else:
                if p_id in top_connect_sents[q_id]:
                    (top_id, top_connect_sent, top_connect_sent_body) = top_connect_sents[q_id][p_id]
                    w1.write(f'{q_id}\t{top_id}\t{qry_collection[q_id]}\t{top_connect_sent_body}\n')
                    w2.write(f'{q_id}\t{top_id}\t{qry_collection[q_id]}\t{top_connect_sent}\n')
                else:
                    print(f'Note: fail to generate connection sentences to target doc {p_id} for query {q_id}.')
                    w1.write(f'{q_id}\t{p_id}\t{qry_collection[q_id]}\t{collection[p_id]}\n')
                    w2.write(f'{q_id}\t{p_id}\t{qry_collection[q_id]}\t"None"\n')


def main():
    parser = ArgumentParser()
    parser.add_argument('--surrogate_model', default=None, required=True)
    parser.add_argument('--surrogate_tag', default=None)
    parser.add_argument('--connect_sent_file', default=None, required=True)
    parser.add_argument('--target_file', default=None, required=True)
    parser.add_argument('--query_collection', default='./msmarco/passage/queries.dev.small.tsv')
    parser.add_argument('--doc_collection', default='./msmarco/passage/collection.tsv')
    parser.add_argument('--coh_weight', type=float, default=0.5)
    parser.add_argument('--rel_weight', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=100)

    args = parser.parse_args()
    print("Run parameters:", args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    qry_collection = {}
    with open(args.query_collection) as f:
        for line in f:
            q_id, q_text = line.strip().split('\t')
            qry_collection[q_id] = q_text.strip()

    collection = {}
    with open(args.doc_collection) as f:
        for line in f:
            d_id, d_text = line.strip().split('\t')
            collection[d_id] = d_text.strip()

    connect_sents = {}
    with open(args.connect_sent_file) as f:
        for line in f:
            q_id, sent_id, q_text, sent_text = line.strip().split('\t')
            p_id = sent_id.split('-')[0]
            if q_id not in connect_sents:
                connect_sents[q_id] = {}
            if p_id not in connect_sents[q_id]:
                connect_sents[q_id][p_id] = {}
            connect_sents[q_id][p_id][sent_id] = sent_text.strip()

    output_nsp_file = args.connect_sent_file + '.position.tt-nsp'
    norm_coh_scores = compute_read_coh_scores(connect_sents=connect_sents, output_nsp_file=output_nsp_file,
                                              collection=collection, device=device, bs=args.batch_size)

    if args.surrogate_tag:
        output_rel_file = args.connect_sent_file + f'.position.{args.surrogate_tag}.rel'
    else:
        output_rel_file = args.connect_sent_file + '.position.rel'
    norm_rel_scores = compute_read_rel_scores(surrogate_model=args.surrogate_model, connect_sents=connect_sents, output_rel_file=output_rel_file,
                                              qry_collection=qry_collection, collection=collection, device=device, bs=args.batch_size)

    if args.surrogate_tag:
        output_qry_sent_file = args.connect_sent_file + f'.position.coh{args.coh_weight}-{args.surrogate_tag}-rel{args.rel_weight}-top1.qry-sent-body.tsv'
        output_qry_sent_body_file = args.connect_sent_file + f'.position.coh{args.coh_weight}-{args.surrogate_tag}-rel{args.rel_weight}-top1.qry-sent.tsv'
    else:
        output_qry_sent_file = args.connect_sent_file + f'.position.coh{args.coh_weight}-rel{args.rel_weight}-top1.qry-sent-body.tsv'
        output_qry_sent_body_file = args.connect_sent_file + f'.position.coh{args.coh_weight}-rel{args.rel_weight}-top1.qry-sent.tsv'

    if os.path.exists(output_qry_sent_file) and os.path.exists(output_qry_sent_body_file):
        print(f'The output files already exist, please check [{output_qry_sent_file}], [{output_qry_sent_body_file}].')
    else:
        assert args.coh_weight + args.rel_weight == 1.0
        merging_connect_sents(connect_sents=connect_sents, collection=collection, qry_collection=qry_collection,
                              norm_coh_scores=norm_coh_scores, norm_rel_scores=norm_rel_scores,
                              target_file=args.target_file, output_qry_sent=output_qry_sent_file,
                              output_qry_sent_body=output_qry_sent_body_file, coh_weight=args.coh_weight,
                              rel_weight=args.rel_weight)

if __name__ == "__main__":
    main()