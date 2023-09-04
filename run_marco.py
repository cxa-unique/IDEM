# Copyright 2021 Reranker Author. All rights reserved.
# Code structure inspired by HuggingFace run_glue.py in the transformers library.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from collections import defaultdict
import torch
import json
from msmarco_mrr_eval import cal_mrr

from reranker import Reranker, RerankerDC
from reranker import RerankerTrainer, RerankerDCTrainer
from reranker.data import GroupedTrainDataset, PredictionDataset, GroupCollator
from reranker.arguments import ModelArguments, DataArguments, \
    RerankerTrainingArguments as TrainingArguments

from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
    set_seed,
)

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    if training_args.training_loss in ['cross_entropy', 'cross_entropy_no_softmax']:
        num_labels = 2
    elif training_args.training_loss in ['hinge', 'LCE', 'hinge_adaptive_score', 'hinge_adaptive_rank']:
        num_labels = 1
    else:
        raise NotImplementedError

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=False,
    )

    _model_class = RerankerDC if training_args.distance_cache else Reranker

    model = _model_class.from_pretrained(
        model_args, data_args, training_args,
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    if training_args.do_train:
        train_dataset = GroupedTrainDataset(
            data_args, data_args.train_path, tokenizer=tokenizer, train_args=training_args
        )
    else:
        train_dataset = None

    # Initialize our Trainer
    _trainer_class = RerankerDCTrainer if training_args.distance_cache else RerankerTrainer
    trainer = _trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=GroupCollator(tokenizer),
    )

    # Training
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_eval:
        trainer.evaluate()

    if training_args.do_predict:
        logging.info("*** Prediction ***")

        if not os.path.exists(data_args.rank_run_path):
            logger.info(f'Creating run directory {data_args.rank_run_path}')
            os.makedirs(data_args.rank_run_path)

        if data_args.only_score:
            rank_run_file = os.path.join(data_args.rank_run_path,
                                         "{0}_scores.tsv".format(os.path.split(data_args.pred_file)[1]))
        else:
            rank_run_file = os.path.join(data_args.rank_run_path,
                                         "{0}_rerank.trec".format(os.path.split(data_args.pred_file)[1]))

        if os.path.exists(rank_run_file):
            raise FileExistsError(f'Run file {rank_run_file} already exists')

        test_dataset = PredictionDataset(
            data_args.pred_file, tokenizer=tokenizer,
            max_len=data_args.max_len,
        )

        pred_qids = []
        pred_pids = []
        with open(data_args.pred_file) as f:
            for l in f:
                item = json.loads(l)
                q, p = item['qid'], item['pid']
                pred_qids.append(q)
                pred_pids.append(p)

        if training_args.training_loss == 'cross_entropy':
            pred_logits = trainer.predict(test_dataset=test_dataset).predictions
            pred_scores = torch.softmax(torch.Tensor(pred_logits), dim=1)[:, 1].detach().cpu().numpy()
        elif training_args.training_loss == 'cross_entropy_no_softmax':
            pred_logits = trainer.predict(test_dataset=test_dataset).predictions
            pred_scores = torch.Tensor(pred_logits)[:, 1].detach().cpu().numpy()

        elif training_args.training_loss in ['hinge', 'LCE']:
            pred_scores = trainer.predict(test_dataset=test_dataset).predictions
        else:
            raise NotImplementedError

        if trainer.is_world_process_zero():
            all_scores = defaultdict(dict)
            assert len(pred_qids) == len(pred_scores)
            for qid, pid, score in zip(pred_qids, pred_pids, pred_scores):
                score = float(score)
                all_scores[qid][pid] = score

            with open(rank_run_file, "w") as writer:
                qq = list(all_scores.keys())
                for qid in qq:
                    if data_args.only_score:
                        for did in all_scores[qid].keys():
                            writer.write(f'{qid}\t{did}\t{all_scores[qid][did]}\n')
                    else:
                        score_list = sorted(list(all_scores[qid].items()), key=lambda x: x[1], reverse=True)
                        for rank, (did, score) in enumerate(score_list):
                            writer.write(f'{qid}\tQ0\t{did}\t{rank + 1}\t{score}\tbert_rerank\n')

            if data_args.qrels_file is not None and not data_args.only_score:
                if 'dev' in data_args.qrels_file:
                    query_num, mrr, mrr_ten, _ = cal_mrr(data_args.qrels_file, rank_run_file)
                    print('-----------------------Metrics------------------------')
                    print(f'query_num: {query_num}, MRR: {mrr}, MRR@10: {mrr_ten}')
                    print('------------------------------------------------------')


if __name__ == "__main__":
    main()
