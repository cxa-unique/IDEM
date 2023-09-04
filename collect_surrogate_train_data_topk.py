import pandas as pd
import collections

ms_data_folder = 'path_to/corpus/msmarco/passage'
runs_data_folder = './msmarco_passage/results/cross-encoder/ms-marco-MiniLM-L-12-v2'
runs_MiniLM_L_12 = runs_data_folder + '/run.msmarco-passage.bm25-default.eval_subset.top1k.features.json_rerank.trec'
save_triples_dir = './msmarco_passage/train_data/surrogate/eval_query/cross-encoder_ms-marco-MiniLM-L-12-v2'

# query_ids = []
# with open(save_triples_dir + '/' + 'run.msmarco-passage.bm25-default.eval_subset-200.query.ids.tsv') as f:
#     for line in f:
#         query_ids.append(line.strip())
# assert len(query_ids) == 200


def collect_from_eval_runs(run_path, save_pre_fix, top_n=29):
    relevant_pairs_dict = collections.defaultdict(list)
    with open(run_path, 'r') as f:
        for line in f:
            qid, _, did, _, _, _ = line.strip().split('\t')
            relevant_pairs_dict[qid].append(did)
            # if qid in query_ids:
            #     relevant_pairs_dict[qid].append(did)

    collected_triples_ids = []
    for seed, (qid, did_list) in enumerate(relevant_pairs_dict.items()):
        # for top n
        tmp_top_n = top_n if len(did_list) > top_n else len(did_list)
        top_n_dids = did_list[:tmp_top_n]
        for i in range(tmp_top_n):
            pos_did = top_n_dids[i]
            for j in range(i + 1, tmp_top_n):
                neg_did = top_n_dids[j]
                collected_triples_ids.append((qid, pos_did, neg_did))

    # save qid pos_did neg_did list
    print("Collect {} triples from: {}".format(len(collected_triples_ids), run_path))

    # load doc_id to string
    collection_df = pd.read_csv("{}/collection.tsv".format(ms_data_folder), sep='\t', names=['docid', 'document_string'])
    collection_df['docid'] = collection_df['docid'].astype(str)
    collection_str = collection_df.set_index('docid').to_dict()['document_string']

    # load query
    query_df = pd.read_csv("{}/queries.eval.subset.tsv".format(ms_data_folder), names=['qid', 'query_string'], sep='\t')
    query_df['qid'] = query_df['qid'].astype(str)
    queries_str = query_df.set_index('qid').to_dict()['query_string']

    sampled_triples_text_list = []
    for (qid, pos_did, neg_did) in collected_triples_ids:
        sampled_triples_text_list.append((qid, pos_did, neg_did, queries_str[qid], collection_str[pos_did], collection_str[neg_did]))

    final_text_triples_df = pd.DataFrame(sampled_triples_text_list)
    save_text_path = save_pre_fix + '.text.top{}.tsv'.format(top_n)
    final_text_triples_df.to_csv(save_text_path, sep='\t', index=False, header=False)
    print("Saved sampled triples text into : {}".format(save_text_path))


if __name__ == "__main__":
    save_triples_prefix = save_triples_dir + '/run.msmarco-passage.bm25-default.eval_subset.top1k.features.json_rerank.trec'
    # save_triples_prefix = save_triples_dir + '/run.msmarco-passage.bm25-default.eval_subset-200.top1k.features.json_rerank.trec'
    collect_from_eval_runs(runs_MiniLM_L_12, save_triples_prefix, top_n=29)