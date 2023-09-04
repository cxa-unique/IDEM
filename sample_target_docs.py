import random
from argparse import ArgumentParser
from collections import defaultdict
import math

def easy_attack_target(initial_rankings, low_rank=50, high_rank=100):
    # sample one target document (being attacked) per 10 candidates
    targets = {}
    for q_id in initial_rankings.keys():
        q_rankings = initial_rankings[q_id]
        rank_len = len(q_rankings)
        if rank_len < low_rank:
            print(f'Note: query {q_id} has only {rank_len} (< {low_rank}) candidates, it will be removed from attack target.')
            continue
        if high_rank > rank_len:
            target_num =  int(math.ceil((rank_len - low_rank) / 10))
        else:
            target_num = int(math.ceil((high_rank - low_rank) / 10))

        if q_id not in targets:
            targets[q_id] = []

        for i in range(target_num):
            p_id, p_rank, p_score = random.sample(q_rankings[low_rank+i*10: low_rank+(i+1)*10], k=1)[0]
            assert low_rank+i*10 < p_rank <= low_rank+(i+1)*10
            targets[q_id].append((p_id, p_rank, p_score))

    return targets


def hard_attack_target(initial_rankings, last_rank=5):
    # select the last ranked candidates
    targets = {}
    for q_id in initial_rankings.keys():
        q_rankings = initial_rankings[q_id]
        rank_len = len(q_rankings)
        if rank_len <= last_rank:
            print(f'Note: query {q_id} has less than 5 candidates, will be removed.')
            continue
        if q_id not in targets:
            targets[q_id] = q_rankings[-last_rank:]
        else:
            raise KeyError

    return targets


def main():
    parser = ArgumentParser()
    parser.add_argument('--rank_file', default='./msmarco_passage/results/cross-encoder/ms-marco-MiniLM-L-12-v2/msmarco-passage.dev.small.1k-query.bm25-default.top1k.features.json_rerank.trec')
    parser.add_argument('--output_target_file', default='./msmarco_passage/attack/cross-encoder_ms-marco-MiniLM-L-12-v2/dev_1k/msmarco-passage.dev.small.1k-query.bm25-default.top1k.rerank.easy-5targets.tsv')
    parser.add_argument('--sample_type', default='easy', help='easy or hard')
    parser.add_argument('--last_rank', type=int, default=5)
    parser.add_argument('--low_rank', type=int, default=50)
    parser.add_argument('--high_rank', type=int, default=100)

    args = parser.parse_args()
    print("Run parameters:", args)

    assert args.rank_file is not None
    assert args.output_target_file is not None

    rankings = defaultdict(list)
    with open(args.rank_file) as f:
        for l in f:
            qid, _, pid, rank, score, _ = l.split()
            rankings[qid].append((pid, int(rank), float(score)))

    if args.sample_type == 'hard':
        targets = hard_attack_target(initial_rankings=rankings, last_rank=args.last_rank)
    elif args.sample_type == 'easy':
        targets = easy_attack_target(initial_rankings=rankings, low_rank=args.low_rank, high_rank=args.high_rank)
    else:
        raise NotImplementedError

    with open(args.output_target_file, 'w') as w:
        for q_id in targets.keys():
            for p_id, p_rank, p_score in targets[q_id]:
                w.write(f'{q_id}\t{p_id}\t{p_rank}\t{p_score}\n')

if __name__ == "__main__":
    main()