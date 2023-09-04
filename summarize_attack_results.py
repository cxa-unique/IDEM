import bisect
from collections import defaultdict
import torch
from tqdm import tqdm
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from argparse import ArgumentParser


class GPT2PPLScorer(object):
    def __init__(self, device):
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

    def perplexity(self, inputs):
        if isinstance(inputs, str):
            inputs = [inputs]
        inputs = self.tokenizer.batch_encode_plus(inputs, padding='longest', return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        # Shift so that tokens < n predict n
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = inputs['input_ids'][:, 1:].contiguous()
        shift_masks = inputs['attention_mask'][:, 1:].contiguous()
        # Flatten the tokens
        loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_labels.shape[0], -1) * shift_masks
        loss = torch.sum(loss, -1) / torch.sum(shift_masks, -1)
        ppl = torch.exp(loss).detach().cpu().numpy().tolist()
        return ppl


def rank_promote_metric(attack_score_file, target_file, target_tag, rank_score_file, rank_list_len=1000):
    attack_scores = defaultdict(dict)
    with open(attack_score_file) as f:
        for line in f:
            qid, did, score = line.strip().split()
            did = did.split('-')[0]
            attack_scores[qid][did] = float(score)

    targets = defaultdict(dict)
    target_num = 0
    with open(target_file) as f:
        for line in f:
            qid, did, rank, score = line.strip().split()
            targets[qid][did] = (int(rank), float(score))
            target_num += 1

    if rank_list_len not in [1000, 100]:
        raise NotImplementedError

    rank_scores = defaultdict(lambda: [-1e6] * rank_list_len)
    with open(rank_score_file) as f:
        for line in f:
            qid, _, did, rank, score, _ = line.strip().split()
            if qid in targets:
                rank_scores[qid][rank_list_len-int(rank)] = float(score)

    rank10 = 0
    rank20 = 0
    rank50 = 0
    rank100 = 0
    avg_boost = 0

    success_num = 0
    same_num = 0
    failure_num = 0
    for q_id in targets:
        old_scores = rank_scores[q_id]
        for d_id in targets[q_id]:
            old_rank, old_score = targets[q_id][d_id]
            assert old_score == old_scores[rank_list_len-old_rank]

            new_score = attack_scores[q_id][d_id]
            new_rank = rank_list_len + 1 - bisect.bisect_right(old_scores, new_score)

            boost = old_rank - new_rank
            if new_rank < old_rank:
                assert new_score > old_score, f'{new_score} vs. {old_score}; {new_rank} vs {old_rank}'
                assert new_score >= old_scores[rank_list_len + 1 - old_rank]
                success_num += 1
            elif new_rank == old_rank:
                assert new_score >= old_score
                if new_score == old_score:
                    print(f'Note target doc {d_id} for query {q_id} has no new score.')
                else:
                    if old_rank != 1:
                        assert new_score < old_scores[rank_list_len+1-old_rank]
                    assert new_score > old_scores[rank_list_len-old_rank]
                same_num += 1
            else:
                assert new_score < old_score
                failure_num += 1
                if rank_list_len == 1000:
                    if 'easy' in target_tag and new_score < old_scores[900]:
                        assert new_rank > 100
                        print(f'Note target doc {d_id} for query {q_id} degrades out from top100.')
                    if 'hard' in target_tag and new_score < old_scores[0]:
                        assert new_rank == 1001
                        print(f'Note target doc {d_id} for query {q_id} degrades out from top1000.')
                else:
                    assert rank_list_len == 100
                    if new_score < old_scores[0]:
                        assert new_rank == 101
                        print(f'Note target doc {d_id} for query {q_id} degrades out from top100.')

            if new_rank <= 10:
                rank10 += 1
            if new_rank <= 20:
                rank20 += 1
            if new_rank <= 50:
                rank50 += 1
            if new_rank <= 100:
                rank100 += 1
            avg_boost += boost

    print('################# Rank Promotion ##################')
    print('target doc: {}'.format(target_num))
    print('success rate: {}'.format(success_num / target_num))
    print('same rate: {}'.format(same_num / target_num))
    print('failure rate: {}'.format(failure_num / target_num))
    print('rank promote <=10: {}'.format(rank10 / target_num))
    print('rank promote <=20: {}'.format(rank20 / target_num))
    print('rank promote <=50: {}'.format(rank50 / target_num))
    print('rank promote <=100: {}'.format(rank100 / target_num))
    print('average boost: {}'.format(avg_boost / target_num))
    print('##################################################')


def compute_ppl_score(text_file, output_ppl_file, device):
    if not os.path.exists(output_ppl_file):
        ppl_scorer = GPT2PPLScorer(device)
        texts = {}
        with open(text_file) as f:
            for line in f:
                q_id, p_id, _, p_text = line.strip().split('\t')
                if q_id not in texts:
                    texts[q_id] = []
                texts[q_id].append((p_id, p_text))

        queries = list(texts.keys())
        total_ppl = []
        with open(output_ppl_file, 'w') as w:
            for q_id in tqdm(queries):
                input_ids = []
                input_texts = []
                for p_id, p_text in texts[q_id]:
                    input_ids.append(p_id)
                    input_texts.append(p_text)
                ppl_list = ppl_scorer.perplexity(inputs=input_texts)
                total_ppl.extend(ppl_list)
                for i, p_id in enumerate(input_ids):
                    w.write(f'{q_id}\t{p_id}\t{ppl_list[i]}\n')
    else:
        total_ppl = []
        with open(output_ppl_file) as f:
            for line in f:
                q_id, p_id, ppl = line.strip().split('\t')
                total_ppl.append(float(ppl))

    print('################# Avg PPL ##################')
    print('target doc: {}'.format(len(total_ppl)))
    print('PPL of GPT2: {}'.format(sum(total_ppl) / len(total_ppl)))
    print('############################################')


def main():
    parser = ArgumentParser()
    parser.add_argument('--device', default='0')
    parser.add_argument('--target_tag', default='easy-5targets')
    parser.add_argument('--target_file', default='./msmarco_passage/attack/cross-encoder_ms-marco-MiniLM-L-12-v2/dev_1k/msmarco-passage.dev.small.1k-query.bm25-default.top1k.rerank.easy-5targets.tsv')
    parser.add_argument('--adv_doc_file', help='the adversarial docs file')
    parser.add_argument('--adv_doc_score_file', help='the rel scores of the adversarial docs')
    parser.add_argument('--rank_score_file', help='the original rel scores of top-1k re-ranked results')
    parser.add_argument('--rank_list_len', type=int, default=1000)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.adv_doc_score_file:
        rank_promote_metric(attack_score_file=args.adv_doc_score_file, target_file=args.target_file,
                            target_tag=args.target_tag, rank_score_file=args.rank_score_file,
                            rank_list_len=args.rank_list_len)

    if args.adv_doc_file:
        output_ppl_file = args.adv_doc_file + '.ppl'
        compute_ppl_score(text_file=args.adv_doc_file, output_ppl_file=output_ppl_file, device=device)


if __name__ == "__main__":
    main()