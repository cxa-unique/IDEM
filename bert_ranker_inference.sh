#!/bin/bash

device=5
train_loss=hinge

### victim NRM
model_dir=cross-encoder/ms-marco-MiniLM-L-12-v2
output_dir=./msmarco_passage/results/${model_dir}

### Surrogate NRM S1
#model_dir=./msmarco_passage/models/surrogate/cross-encoder_ms-marco-MiniLM-L-12-v2/eval_query/bert_base_eval-top1k-rerank_top29_hinge_margin1_ml484_lr3e-6_bs16_ep2
#output_dir=./msmarco_passage/results/surrogate/cross-encoder_ms-marco-MiniLM-L-12-v2/eval_query/bert_base_eval-top1k-rerank_top29_hinge_margin1_ml484_lr3e-6_bs16_ep2

### Surrogate NRM S2
#model_dir=./msmarco_passage/models/surrogate/cross-encoder_ms-marco-MiniLM-L-12-v2/eval_query/bert_base_eval-200-top1k-rerank_top29_hinge_margin1_ml484_lr3e-6_bs16_ep2
#output_dir=./msmarco_passage/results/surrogate/cross-encoder_ms-marco-MiniLM-L-12-v2/eval_query/bert_base_eval-200-top1k-rerank_top29_hinge_margin1_ml484_lr3e-6_bs16_ep2

### Surrogate NRM S3
#model_dir=./dpr_nq/models/bert_base_hinge_margin1_ml484_lr3e-6_bs16_p1n8-ep1
#output_dir=./dpr_nq/results/bert_base_hinge_margin1_ml484_lr3e-6_bs16_p1n8-ep1

### Eval top-1k BM25
pred_file=./msmarco_passage/train_data/surrogate/eval_query/run.msmarco-passage.bm25-default.eval_subset.top1k.features.json

### Dev top-1k BM25
#pred_file=./msmarco_passage/test_data/msmarco-passage.dev.small.run.bm25-default.top1k.features.json
#qrels=/home1/cxa/ir_data/corpus/msmarco/passage/qrels.dev.small.tsv

CUDA_VISIBLE_DEVICES=${device} python run_marco.py \
                              --output_dir=temp \
                              --model_name_or_path ${model_dir} \
                              --do_predict \
                              --max_len 512 \
                              --per_device_eval_batch_size 100 \
                              --dataloader_num_workers 8 \
                              --pred_file ${pred_file} \
                              --rank_run_path ${output_dir} \
                              --training_loss ${train_loss} \
#                              --qrels_file ${qrels}
