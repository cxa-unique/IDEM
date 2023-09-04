# IDEM
This repository contains the code and resources for our paper:

Xuanang Chen, Ben He, Zheng Ye, Le Sun, Yingfei Sun: 
[Towards Imperceptible Document Manipulations against Neural Ranking Models](https://aclanthology.org/2023.findings-acl.416/).
In *Findings of ACL 2023*.

<img src=https://github.com/cxa-unique/IDEM/blob/main/idem_framework.png width=60% />


## Installation
Our training code is developed based on [Reranker](https://github.com/luyug/Reranker) training toolkit. 
We recommend you to create a new conda environment `conda create -n idem python=3.7`, activate it `conda activate rodr`, 
and then install the following packages: `torch==1.8.1`, `transformers==4.9.2`, `datasets==1.11.0`, `spacy==3.4.1`.
A few open toolkits are also needed, including `pytorch/fairseq` and `rbo`. 


## Getting Started
### 1. Surrogate NRMs
This work focuses on the decision-based black-box attack setting, it is needed to collect pair-wise training samples from
the output ranking lists of victim NRM, which are used to train a surrogate NRM.
The '[msmarco-MiniLM-L-12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-12-v2)' model publicly available at 
*Hugging Face* is used as the representative victim NRM in our study.

**a) Data Preparation:** 
Given a set of queries (e.g., 6,837 Eval queries in MS MARCO passage dataset) and their top-1K BM25 candidates by [Anserini]()
in the format of `query_id \t doc_id \t query_text \t doc_text \n`, we can convert the raw text into the input features 
(refer to `topk_text_2_json.py`), and run re-ranking step to obtain the re-ranking results of the victim NRM 
(refer to `bert_ranker_inference.sh`).
After that, we can collect paired training samples from the top-k results (refer to `collect_surrogate_train_data_topk.py`), and 
convert each training sample into features (refer to `train_text_2_json.py`), in the format of the following example:
```
{"qry": {"qid": 57, "query": [2744, 2326, 3820, 6210]}, 
 "pos": {"pid": 4537032, "passage": [1037, 2326, 3820, 2003, 1037, 5337, 3820, 2090, ...]}, 
 "neg": {"pid": 5745108, "passage": [18875, 1037, 3408, 1997, 2326, 1012, 2976, 1011, ...]}}
```
Note: when the access to victim model is limited or unavailable, we need to reduce the number of Eval queries (200 in our work)
or use the Natural Question dataset (`biencoder-nq-train.json` released in [DPR](https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py#L45) 
for training instead (refer `collect_surrogate_train_data_nq.py`).

**b) Model Training:**
The surrogate NRMs are based on the pre-trained BERT-Base model, and we directly adopt the hinge loss with a margin of 1 
for ranking imitation (refer to `bert_ranker_train.sh`).
After training, we can evaluate the performance of the surrogate NRMs on the Dev set in MS MARCO passage dataset (refer to `bert_ranker_inference.sh`),
and can also compute the ranking similarities (like the overlap of top-10 and the [Rank Biased Overlap](https://github.com/changyaochen/rbo)) to measure the 
ranking consistency between the surrogate and victim NRMs (refer to `compute_ranking_similarities.py`).


### 2. Document Manipulations
We evaluate attack methods on randomly sampled 1K Dev queries (in [Resources](https://github.com/cxa-unique/IDEM/#Resources)) 
with two types of target documents, i.e., Easy-5 and Hard-5, which are sampled from the re-ranked results by the victim 
NRM on the top1K BM25 candidates, please refer to `sample_target_docs.py`.

**Stage 1: connection sentences generation:**\
Given a query *q* and a target document *d*, we concatenate them with a blank in between, and engineer generative language
models in a blank-infilling pattern to generate connection sentences that include information about both the query and the document.
```
CUDA_VISIBLE_DEVICES=0 python -u generate_connection_sents.py \
                                  --model_name bart.base \
                                  --target_file {the file contains target documents} \
                                  --query_collection './msmarco/passage/queries.dev.small.tsv' \
                                  --doc_collection './msmarco/passage/collection.tsv' \
                                  --prompt_str "It is known that" \
                                  --connect_sent_max_len 12 \
                                  --connect_sent_num 100 \
                                  --sample_size 50 \
                                  --max_sample_times 10 \
                                  --sampling_topk 50 \
                                  --output_connect_sents {the output file contains generated connection sentences}

```
BART-Base model is used to generate connection sentences by default. 
We can change the prefix text by assigning `--prompt_str`, and adjust the number or length of connection sentences by setting 
`--connect_sent_num` or `--connect_sent_max_len`.


**Stage 2: merging with original document:**\
To balance between the attack effect and the fidelity of perturbed document content, we design a position-wise merging 
strategy to place an appropriate connection sentence at an optimal position within the original target document.
```
CUDA_VISIBLE_DEVICES=0 python -u merge_connection_sents_with_docs.py \
                              --surrogate_model {the folder contains the surrogate model} \
                              --surrogate_tag {the tag of the surrogate model} \
                              --connect_sent_file {the file contains generated connection sentences} \
                              --target_file {the file contains target documents} \
                              --query_collection './msmarco/passage/queries.dev.small.tsv' \
                              --doc_collection './msmarco/passage/collection.tsv' \
                              --coh_weight 0.5 \
                              --rel_weight 0.5 \
                              --batch_size 100 \
```
We can specify which surrogate NRM is used to evaluate the pseudo-relevance of adversarial documents by setting the arguments
`--surrogate_model` and `--surrogate_tag`. There are two output files in the same folder for the `connect_sent_file`, 
their names end with `qry-sent-body.tsv` and `qry-sent.tsv`, respectively. The `qry-sent-body.tsv` file contains the
final adversarial documents, and the `qry-sent.tsv` file contains the selected connection sentences.


### 3. Attack Results
After generating the adversarial documents, we would like to see how the outcome of the attack is. 
At first, we should convert the adversarial document text into features by running `topk_text_2_json.py`, and then we feed
them into the victim NRM to get the real relevance scores by running `run_marco.py`, and finally we summarize the attack
results by running `summarize_attack_results.py`. Please refer to `summarize_idem_attack_results.sh` as an example.

## Resources
1. Sampled 1K Dev queries: `data/msmarco-passage.dev.small.attack.1000-query-id.txt`

2. Sampled target documents: 
    - Ranking list: `data/msmarco-passage.dev.small.1k-query.bm25-default.top1k.features.json_rerank.trec`
    - Easy-5: `data/msmarco-passage.dev.small.1k-query.bm25-default.top1k.rerank.easy-5targets.tsv`
    - Hard-5: `data/msmarco-passage.dev.small.1k-query.bm25-default.top1k.rerank.hard-5targets.tsv`

3. Generated connection sentences and adversarial documents:

    |  Method  |  Easy-5  |  Hard-5  | 
    |----------|----------|----------|
    | IDEM-100 | [Download](https://drive.google.com/file/d/1VxcZ9rLvBxP3Y-A-E9P4yDDPqM-pqG9g/view?usp=sharing) | [Download](https://drive.google.com/file/d/17k0DBYD_aguTKYjc52uX0H5dW20dPfn4/view?usp=sharing) |
    | IDEM-500 | [Download](https://drive.google.com/file/d/1nPh3Yh9QVI7lDbx1K2qex3EW0ZmyuIlQ/view?usp=sharing) | [Download](https://drive.google.com/file/d/1gno8m2isgZ8QZYdUMl43RSZpIUtW_of2/view?usp=sharing) |
      
    - Each compressed folder contain 7 files, one for all connection sentences, two for selected connection sentences and 
    final adversarial documents under surrogate NRM S1 (`all-top29-ep2`), S2 (`200-top29-ep2`) and S3 (`nq-p1n8-ep1`), respectively.
    
    - Let's take **IDEM-100_Easy-5** and **S1** as an example:
    1) all connection sentences: `msmarco-passage.dev.small.1k-query.bm25-default.top1k.rerank.easy-5targets.query-iikt-mask-body.bart.base.top50sampling.len12-num100.tsv`    
    2) selected connection sentences under S1: `msmarco-passage.dev.small.1k-query.bm25-default.top1k.rerank.easy-5targets.query-iikt-mask-body.bart.base.top50sampling.len12-num100.tsv.position.coh0.5-all-top29-ep2-rel0.5-top1.qry-sent.tsv`
    3) final adversarial documents under S1: `msmarco-passage.dev.small.1k-query.bm25-default.top1k.rerank.easy-5targets.query-iikt-mask-body.bart.base.top50sampling.len12-num100.tsv.position.coh0.5-all-top29-ep2-rel0.5-top1.qry-sent-body.tsv`
    
   
4. Surrogate NRMs: [S1](https://drive.google.com/file/d/1WR7ZARWrZJZParHL1s_S-t6X-mzBwBhd/view?usp=sharing), [S2](https://drive.google.com/file/d/1R176s7NKLy6UQHwQxMJCHoG_AVrmYCHn/view?usp=sharing), and [S3](https://drive.google.com/file/d/1w3y19XnwfqZkV3ELy5KEaralzqI1wbK6/view?usp=sharing).


## Citation
If you find our paper/resources useful, please cite:
```
@inproceedings{DBLP:conf/acl/ChenHY0S23,
  author       = {Xuanang Chen and
                  Ben He and
                  Zheng Ye and
                  Le Sun and
                  Yingfei Sun},
  title        = {Towards Imperceptible Document Manipulations against Neural Ranking Models},
  booktitle    = {Findings of the Association for Computational Linguistics: {ACL} 2023},
  pages        = {6648--6664},
  publisher    = {Association for Computational Linguistics},
  year         = {2023},
  url          = {https://doi.org/10.18653/v1/2023.findings-acl.416},
  doi          = {10.18653/v1/2023.findings-acl.416},
}
```
