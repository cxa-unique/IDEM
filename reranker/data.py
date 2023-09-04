# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
from dataclasses import dataclass

import datasets
from typing import Union, List, Tuple, Dict

import torch
from torch.utils.data import Dataset

from .arguments import DataArguments, RerankerTrainingArguments
from transformers import PreTrainedTokenizer, BatchEncoding
from transformers import DataCollatorWithPadding


class GroupedTrainDataset(Dataset):
    query_columns = ['qid', 'query']
    document_columns = ['pid', 'passage']

    def __init__(
            self,
            args: DataArguments,
            path_to_tsv: Union[List[str], str],
            tokenizer: PreTrainedTokenizer,
            train_args: RerankerTrainingArguments = None,
    ):
        if train_args.training_loss == 'LCE':
            self.nlp_dataset = datasets.load_dataset(
                'json',
                data_files=path_to_tsv,
                ignore_verifications=False,
                features=datasets.Features({
                    'qry': {
                        'qid': datasets.Value('string'),
                        'query': [datasets.Value('int32')],
                    },
                    'pos': [{
                        'pid': datasets.Value('string'),
                        'passage': [datasets.Value('int32')],
                    }],
                    'neg': [{
                        'pid': datasets.Value('string'),
                        'passage': [datasets.Value('int32')],
                    }]}
                )
            )['train']
        elif train_args.training_loss == 'cross_entropy':
            self.nlp_dataset = datasets.load_dataset(
                'json',
                data_files=path_to_tsv,
                ignore_verifications=False,
                features=datasets.Features({
                    'qry': {
                        'qid': datasets.Value('string'),
                        'query': [datasets.Value('int32')],
                    },
                    'text': {
                        'pid': datasets.Value('string'),
                        'passage': [datasets.Value('int32')],
                    },
                    'label': datasets.Value('int32')}
                )
            )['train']
        elif train_args.training_loss == 'hinge' or train_args.training_loss == 'hinge_adaptive_score':
            self.nlp_dataset = datasets.load_dataset(
                'json',
                data_files=path_to_tsv,
                ignore_verifications=False,
                features=datasets.Features({
                    'qry': {
                        'qid': datasets.Value('string'),
                        'query': [datasets.Value('int32')],
                    },
                    'pos': {
                        'pid': datasets.Value('string'),
                        'passage': [datasets.Value('int32')],
                    },
                    'neg': {
                        'pid': datasets.Value('string'),
                        'passage': [datasets.Value('int32')],
                    }}
                )
            )['train']
        elif train_args.training_loss == 'hinge_adaptive_rank':
            self.nlp_dataset = datasets.load_dataset(
                'json',
                data_files=path_to_tsv,
                ignore_verifications=False,
                features=datasets.Features({
                    'qry': {
                        'qid': datasets.Value('string'),
                        'query': [datasets.Value('int32')],
                    },
                    'pos': {
                        'pid': datasets.Value('string'),
                        'passage': [datasets.Value('int32')],
                    },
                    'neg': {
                        'pid': datasets.Value('string'),
                        'passage': [datasets.Value('int32')],
                    },
                    'gap': datasets.Value('int32')}
                )
            )['train']

        else:
            raise NotImplementedError

        self.tok = tokenizer
        self.SEP = [self.tok.sep_token_id]
        self.args = args
        self.total_len = len(self.nlp_dataset)
        self.train_args = train_args

        if train_args is not None and train_args.collaborative:
            import torch.distributed as dist
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
            chunk_size = int(self.args.train_group_size / self.world_size)
            self.chunk_start = self.rank * chunk_size
            self.chunk_end = self.chunk_start + chunk_size

    def create_one_example(self, qry_encoding: List[int], doc_encoding: List[int]):
        item = self.tok.encode_plus(
            qry_encoding,
            doc_encoding,
            truncation='only_second',
            max_length=self.args.max_len,
            padding=False,
        )
        return item

    def create_one_example_label(self, qry_encoding: List[int], doc_encoding: List[int], label_id: int):
        item = self.tok.encode_plus(
            qry_encoding,
            doc_encoding,
            truncation='only_second',
            max_length=self.args.max_len,
            padding=False,
        )
        # label_id = torch.tensor(label_id, dtype=torch.long)
        item['labels'] = label_id
        return item

    def create_one_example_gap(self, qry_encoding: List[int], doc_encoding: List[int], rank_gap: int):
        item = self.tok.encode_plus(
            qry_encoding,
            doc_encoding,
            truncation='only_second',
            max_length=self.args.max_len,
            padding=False,
        )
        item['rank_gap'] = rank_gap
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> List[BatchEncoding]:
        group = self.nlp_dataset[item]

        if self.train_args.training_loss == 'LCE':
            examples = []
            group_batch = []
            _, qry = (group['qry'][k] for k in self.query_columns)
            _, pos_psg = [
                random.choice(group['pos'])[k] for k in self.document_columns]
            examples.append((qry, pos_psg))

            if len(group['neg']) < self.args.train_group_size - 1:
                negs = random.choices(group['neg'], k=self.args.train_group_size - 1)
            else:
                negs = random.sample(group['neg'], k=self.args.train_group_size - 1)

            for neg_entry in negs:
                _, neg_psg = [neg_entry[k] for k in self.document_columns]
                examples.append((qry, neg_psg))

            # collaborative mode, split the group
            if self.train_args is not None and self.train_args.collaborative:
                examples = examples[self.chunk_start: self.chunk_end]

            for e in examples:
                group_batch.append(self.create_one_example(*e))
            return group_batch

        elif self.train_args.training_loss == 'cross_entropy':
            group_batch = []
            _, qry = (group['qry'][k] for k in self.query_columns)
            _, psg = (group['text'][k] for k in self.document_columns)
            label = group['label']
            group_batch.append(self.create_one_example_label(qry, psg, label))
            return group_batch

        elif self.train_args.training_loss == 'hinge' or self.train_args.training_loss == 'hinge_adaptive_score':
            group_batch = []
            _, qry = (group['qry'][k] for k in self.query_columns)
            _, pos = (group['pos'][k] for k in self.document_columns)
            _, neg = (group['neg'][k] for k in self.document_columns)
            group_batch.append(self.create_one_example(qry, pos))
            group_batch.append(self.create_one_example(qry, neg))
            return group_batch

        elif self.train_args.training_loss == 'hinge_adaptive_rank':
            group_batch = []
            _, qry = (group['qry'][k] for k in self.query_columns)
            _, pos = (group['pos'][k] for k in self.document_columns)
            _, neg = (group['neg'][k] for k in self.document_columns)
            rank_gap = group['gap']
            group_batch.append(self.create_one_example_gap(qry, pos, rank_gap))
            group_batch.append(self.create_one_example_gap(qry, neg, rank_gap))
            return group_batch
        else:
            raise NotImplementedError


class PredictionDataset(Dataset):
    columns = [
        'qid', 'pid', 'qry', 'psg'
    ]

    def __init__(self, pred_json_file: str, tokenizer: PreTrainedTokenizer, max_len=128):
        self.nlp_dataset = datasets.load_dataset(
            'json',
            data_files=[pred_json_file],
        )['train']
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.nlp_dataset)

    def __getitem__(self, item):
        qid, pid, qry, psg = (self.nlp_dataset[item][f] for f in self.columns)
        return self.tok.encode_plus(
            qry,
            psg,
            truncation='only_second',
            max_length=self.max_len,
            padding=False,
        )


@dataclass
class GroupCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    def __call__(
            self, features
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if isinstance(features[0], list):
            features = sum(features, [])
        return super().__call__(features)
