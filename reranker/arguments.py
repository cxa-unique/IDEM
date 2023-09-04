# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass, field
from typing import Optional, Union, List
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    temperature: Optional[float] = field(default=None)


@dataclass
class DataArguments:
    train_dir: str = field(
        default=None, metadata={"help": "Path to train directory"}
    )
    train_path: Union[str] = field(
        default=None, metadata={"help": "Path to train data"}
    )
    train_group_size: int = field(default=8)
    dev_path: str = field(
        default=None, metadata={"help": "Path to dev data"}
    )
    pred_file: str = field(default=None, metadata={"help": "Prediction file"})
    only_score: bool = field(default=False, metadata={"help": "whether to only save the match score"})
    rank_run_path: str = field(default=None, metadata={"help": "where to save the rank result file"})
    max_len: int = field(
        default=128,
        metadata={"help": "The maximum total input sequence length after tokenization for passage. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."})
    qrels_file: str = field(default=None, metadata={"help": "the golden label file"})

    def __post_init__(self):
        if self.train_dir is not None:
            files = os.listdir(self.train_dir)
            self.train_path = [
                os.path.join(self.train_dir, f)
                for f in files
                if f.endswith('tsv') or f.endswith('json')
            ]


@dataclass
class RerankerTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    distance_cache: bool = field(default=False)
    distance_cache_stride: int = field(default=2)

    collaborative: bool = field(default=False)
    curriculum: bool = field(default=False)

    training_loss: str = field(default=None, metadata={"help": "Training loss"})
