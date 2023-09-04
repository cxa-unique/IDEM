#!/bin/bash

train_loss=hinge
learning_rate=3e-6
batch_size=16
max_len=484
device=5

#### Surrogate NRM S1
save_steps=10000
train_epoch=2
train_data=./msmarco_passage/train_data/surrogate/eval_query/cross-encoder_ms-marco-MiniLM-L-12-v2/run.msmarco-passage.bm25-default.eval_subset.top1k.features.json_rerank.trec.top29last0.json
output_dir=./msmarco_passage/models/surrogate/cross-encoder_ms-marco-MiniLM-L-12-v2/eval_query/bert_base_eval-top1k-rerank_top29last0_${train_loss}_margin1_ml${max_len}_lr${learning_rate}_bs${batch_size}_ep${train_epoch}

#### Surrogate NRM S2
#q_num=200
#save_steps=5000
#train_epoch=2
#train_data=./msmarco_passage/train_data/surrogate/eval_query/cross-encoder_ms-marco-MiniLM-L-12-v2/run.msmarco-passage.bm25-default.eval_subset-${q_num}.top1k.features.json_rerank.trec.top29.json
#output_dir=./msmarco_passage/models/surrogate/cross-encoder_ms-marco-MiniLM-L-12-v2/eval_query/bert_base_eval-${q_num}-top1k-rerank_top29_${train_loss}_margin1_ml${max_len}_lr${learning_rate}_bs${batch_size}_ep${train_epoch}

#### Surrogate NRM S3
#save_steps=10000
#train_epoch=1
#train_data=./dpr_nq/train_data/biencoder-nq-train-triples-p1n8_hinge.json
#output_dir=./dpr_nq/models/bert_base_${train_loss}_margin1_ml${max_len}_lr${learning_rate}_bs${batch_size}_p1n8-ep${train_epoch}


model_name_or_path=bert-base-uncased
CUDA_VISIBLE_DEVICES=${device} python run_marco.py \
                              --output_dir ${output_dir} \
                              --model_name_or_path ${model_name_or_path} \
                              --do_train \
                              --save_steps ${save_steps} \
                              --train_path ${train_data} \
                              --max_len ${max_len} \
                              --fp16 \
                              --per_device_train_batch_size ${batch_size} \
                              --gradient_accumulation_steps 1 \
                              --warmup_ratio 0.1 \
                              --weight_decay 0.01 \
                              --learning_rate ${learning_rate} \
                              --num_train_epochs ${train_epoch} \
                              --overwrite_output_dir \
                              --dataloader_num_workers 8 \
                              --training_loss ${train_loss}