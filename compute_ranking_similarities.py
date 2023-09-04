import rbo

rank_file = 'msmarco-passage.dev.small.run.bm25-default.top1k.features.json_rerank.trec'
pre_file1 = f'./msmarco_passage/results/cross-encoder/ms-marco-MiniLM-L-12-v2/{rank_file}'

model_name = 'bert_base_eval-top1k-rerank_top29last0_hinge_margin1_ml484_lr3e-6_bs16_ep2'
# model_name = 'bert_base_eval-200-top1k-rerank_top29last0_hinge_margin1_ml484_lr3e-6_bs16_ep2'
pre_file2 = f'./msmarco_passage/results/surrogate/cross-encoder_ms-marco-MiniLM-L-12-v2/eval_query/{model_name}/{rank_file}'
# pre_file2 = f'./dpr_nq/results/bert_base_hinge_margin1_ml484_lr3e-6_bs16_p1n8-ep1/{rank_file}'

res1 = {}
with open(pre_file1, 'r') as file:
    for line in file:
        [qid, _, docid, rank, _, _] = line.split()
        if qid not in res1:
            res1[qid] = []
        res1[qid].append(docid)

res2 = {}
with open(pre_file2, 'r') as file:
    for line in file:
        [qid, _, docid, rank, _, _] = line.split()
        if qid not in res2:
            res2[qid] = []
        res2[qid].append(docid)


rbo_p = 0.7
overlap_cut = 10
rbo_per_query = []
overlap_per_query = []

for q_id in res1.keys():
    S = res1[q_id]
    T = res2[q_id]
    assert len(S) == len(T)
    rbo_per_query.append(rbo.RankingSimilarity(S, T).rbo(p=rbo_p))

    overlap = [x for x in S[:overlap_cut] if x in T[:overlap_cut]]
    overlap_per_query.append(len(overlap) / 10)

assert len(rbo_per_query) == 6980
assert len(overlap_per_query) == 6980

print(f'RBO@1K: {sum(rbo_per_query) / len(rbo_per_query)}')
print(f'Inter@10: {sum(overlap_per_query) / len(overlap_per_query)}')