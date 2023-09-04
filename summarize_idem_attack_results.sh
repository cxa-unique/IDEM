#!/bin/bash

device=2
target_tag=easy-5targets
#target_tag=hard-5targets

prefix_dir=prefix_it_is_known_that
template_tag=query-iikt-mask-body
run_prefix=msmarco-passage.dev.small.1k-query.bm25-default.top1k.rerank

model=bart.base
max_len=12
dir_prefix=./msmarco_passage/attack/cross-encoder_ms-marco-MiniLM-L-12-v2/dev_1k/${model}/${prefix_dir}/topk50_ss50_st10
generate_tag=${model}.top50sampling.len${max_len}-num100
#dir_prefix=./msmarco_passage/attack/cross-encoder_ms-marco-MiniLM-L-12-v2/dev_1k/${model}/${prefix_dir}/topk50_ss50_st50
#generate_tag=${model}.top50sampling.len${max_len}-num500

#### Surrogate NRM S1
#surrogate_model=./msmarco_passage/models/surrogate/cross-encoder_ms-marco-MiniLM-L-12-v2/eval_query/bert_base_eval-top1k-rerank_top29_hinge_margin1_ml484_lr3e-6_bs16_ep2
#surrogate_tag=all-top29-ep2

#### Surrogate NRM S2
#surrogate_model=./msmarco_passage/models/surrogate/cross-encoder_ms-marco-MiniLM-L-12-v2/eval_query/bert_base_eval-200-top1k-rerank_top29_hinge_margin1_ml484_lr3e-6_bs16_ep2
#surrogate_tag=200-top29-ep2

#### Surrogate NRM S3
#surrogate_model=./dpr_nq/models/bert_base_hinge_margin1_ml484_lr3e-6_bs16_p1n8-ep1
#surrogate_tag=nq-p1n8-ep1

coh_weight=0.5
rel_weight=0.5
#coh_weight=0.1
#rel_weight=0.9

generate_tag=${generate_tag}.tsv.position.coh${coh_weight}-rel${rel_weight}-top1
#generate_tag=${generate_tag}.tsv.position.coh${coh_weight}-${surrogate_tag}-rel${rel_weight}-top1

### convert text to features
adv_doc_file=${dir_prefix}/${run_prefix}.${target_tag}.${template_tag}.${generate_tag}.qry-sent-body.tsv
save_to_file=${dir_prefix}/${run_prefix}.${target_tag}.${template_tag}.${generate_tag}.qry-sent-body.features.json

python topk_text_2_json.py --file ${adv_doc_file} \
                           --save_to ${save_to_file} \
                           --tokenizer bert-base-uncased \
                           --truncate 512 \
                           --q_truncate -1

### inference relevance scores after attack
train_loss=hinge
model_dir=cross-encoder/ms-marco-MiniLM-L-12-v2

CUDA_VISIBLE_DEVICES=${device} python run_marco.py \
                              --output_dir=temp \
                              --model_name_or_path ${model_dir} \
                              --do_predict \
                              --max_len 512 \
                              --per_device_eval_batch_size 100 \
                              --dataloader_num_workers 8 \
                              --pred_file ${save_to_file} \
                              --rank_run_path ${dir_prefix} \
                              --training_loss ${train_loss} \
                              --only_score

### summarize attack results
ori_rank_score_file=./msmarco_passage/results/cross-encoder/ms-marco-MiniLM-L-12-v2/msmarco-passage.dev.small.1k-query.bm25-default.top1k.features.json_rerank.trec
target_file=./msmarco_passage/attack/cross-encoder_ms-marco-MiniLM-L-12-v2/dev_1k/${run_prefix}.${target_tag}.tsv
adv_doc_score_file=${save_to_file}_scores.tsv

python summarize_attack_results.py --device ${device} \
                                   --target_tag ${target_tag} \
                                   --target_file ${target_file} \
                                   --adv_doc_file ${adv_doc_file} \
                                   --adv_doc_score_file ${adv_doc_score_file} \
                                   --rank_score_file ${ori_rank_score_file} \
                                   --rank_list_len 1000
