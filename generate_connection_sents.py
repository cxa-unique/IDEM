import torch
from argparse import ArgumentParser
import random
from tqdm import tqdm
from collections import defaultdict
import spacy
import string
import difflib
import os


punc = string.punctuation
nlp = spacy.load('en_core_web_sm')


def preprocess_query(query_text):
    query_text = query_text.strip()
    query_words = query_text.split()
    if query_text[-1] not in punc:
        first_word = query_words[0]
        if "'" in first_word:
            first_word = first_word.split("'")[0]
        if first_word.lower() in ['what', 'who', 'which', 'whose', 'whom', 'which', 'whose', 'when', 'where', 'why', 'how']:
            query_text = query_text + '?'
        else:
            query_text = query_text + '.'
    assert query_text[-1] in punc
    return query_text


def filling_checker(q_text, p_body, span, max_word_len=12, max_match_radio=0.5):
    span = span.lower()
    q_text = q_text.lower()
    p_body = p_body.lower()

    if len(span) < 1 or span in punc:
        return False
    if q_text.lower() in span.lower():  ## assert not copy query
        return False
    if len(span.split()) > max_word_len:  ## the max word num
        return False

    match_radio_query = difflib.SequenceMatcher(None, span, q_text).find_longest_match(0, len(span), 0, len(q_text)).size / len(span)
    match_radio_body = difflib.SequenceMatcher(None, span, p_body).find_longest_match(0, len(span), 0, len(p_body)).size / len(span)
    if match_radio_query > max_match_radio or match_radio_body > max_match_radio:
        return False

    return True


prompt_list = ['We know that', 'It is about that', 'It is known that', 'The fact is that']


def filling_generation(args, model, q_id, q_text, p_ids, prompts, contexts, collection, writer):

    fillings = {}
    for p_id in p_ids:
        fillings[p_id] = []
    gen_index = [i for i in range(len(p_ids))]

    for _ in range(args.max_sample_times):
        if len(gen_index) > 0:
            gen_p_ids = [p_ids[j] for j in gen_index]
            gen_prompts = [prompts[j] for j in gen_index]
            gen_contexts = [contexts[j] for j in gen_index]

            with torch.no_grad():
                gen_texts = model.fill_mask(gen_contexts, topk=args.sample_size, sampling=True,
                                            sampling_topk=args.sampling_topk, match_source_len=False)
                torch.cuda.empty_cache()

            for k, gen_text in enumerate(gen_texts):
                p_id = gen_p_ids[k]
                prompt = gen_prompts[k]
                p_body = collection[p_id]
                assert len(gen_text) == args.sample_size

                for (gen_body, _) in gen_text:
                    gen_body_sents = [str(s).strip() for s in nlp(gen_body).sents]
                    for sent in gen_body_sents:
                        if sent[:len(prompt)] == prompt and sent not in p_body:
                            span = sent[len(prompt):].strip()  ## delete the prompt
                            if not filling_checker(q_text, p_body, span, max_word_len=args.connect_sent_max_len):
                                continue
                            if span[0].islower():
                                span = span[0].upper() + span[1:]
                            if span not in fillings[p_id] and len(fillings[p_id]) < args.connect_sent_num:
                                fillings[p_id].append(span)

        gen_index = [i for i, pid in enumerate(p_ids) if len(fillings[pid]) < args.connect_sent_num]

    for p_id in fillings.keys():
        if len(fillings[p_id]) == 0:
            print(f'Note: there is no valid filling generated for query {q_id} and doc {p_id}.')
            continue
        if 0 < len(fillings[p_id]) < args.connect_sent_num:
            print(f'Note: there is only {len(fillings[p_id])} < {args.connect_sent_num} fillings '
                  f'generated for query {q_id} and doc {p_id}.')

        for k, valid_span in enumerate(fillings[p_id]):
            fp_id = p_id + f'-{k}'
            writer.write(f'{q_id}\t{fp_id}\t{q_text}\t{valid_span}\n')


def connect_sent_generate(args, model, targets, qry_collection, collection, output_file, prompt_str, max_inference_size=5):
    queries = list(targets.keys())
    with open(output_file, 'w') as writer:
        for q_id in tqdm(queries):
            q_text = qry_collection[q_id]
            q_text_punc = preprocess_query(q_text)

            contexts = []
            p_ids = []
            prompts = []
            for (p_id, _, _) in targets[q_id]:
                p_body = collection[p_id]
                if prompt_str == 'Random':
                    prompt = random.sample(prompt_list, k=1)[0]
                else:
                    assert prompt_str in prompt_list
                    prompt = prompt_str

                if prompt in p_body:
                    print(f'Please note that prompt {prompt} exists in doc {p_id}.')

                context = q_text_punc + f' {prompt} <mask> ' + p_body

                contexts.append(context)
                p_ids.append(p_id)
                prompts.append(prompt)

                #### inference once per a batch
                if len(p_ids) == max_inference_size:
                    filling_generation(args, model, q_id, q_text, p_ids, prompts, contexts, collection, writer=writer)
                    contexts = []
                    p_ids = []
                    prompts = []

            #### inference for the rest targets
            if len(p_ids) > 0:
                filling_generation(args, model, q_id, q_text, p_ids, prompts, contexts, collection, writer=writer)


def main():
    parser = ArgumentParser()
    parser.add_argument('--model_name', default='bart.base')
    parser.add_argument('--target_file', required=True)
    parser.add_argument('--query_collection', required=True)
    parser.add_argument('--doc_collection', required=True)
    parser.add_argument('--prompt_str', type=str, default='It is known that')
    parser.add_argument('--connect_sent_max_len', type=int, default=12)
    parser.add_argument('--connect_sent_num', type=int, default=100)
    parser.add_argument('--sample_size', type=int, default=50)
    parser.add_argument('--max_sample_times', type=int, default=10)
    parser.add_argument('--sampling_topk', type=int, default=50)
    parser.add_argument('--output_connect_sents', required=True)

    args = parser.parse_args()
    print("Run parameters:", args)
    assert args.max_sample_times * args.sample_size > args.filling_num

    assert args.target_file is not None
    targets = defaultdict(list)
    with open(args.target_file) as f:
        for l in f:
            qid, pid, rank, score = l.split()
            targets[qid].append((pid, int(rank), float(score)))

    output_path, _ = os.path.split(args.output_connect_sents)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if os.path.exists(args.output_connect_sents) and os.path.getsize(args.output_connect_sents):
        raise FileExistsError

    collection = {}
    with open(args.doc_collection) as f:
        for line in f:
            d_id, d_text = line.strip().split('\t')
            collection[d_id] = d_text.strip()

    qry_collection = {}
    with open(args.query_collection) as f:
        for line in f:
            q_id, q_text = line.strip().split('\t')
            qry_collection[q_id] = q_text.strip()

    # prepare mask filling model
    BART = torch.hub.load('pytorch/fairseq', args.model_name)
    BART.cuda()
    BART.eval()

    connect_sent_generate(args, model=BART, targets=targets, qry_collection=qry_collection, collection=collection,
                          output_file=args.output_connect_sents, prompt_str=args.prompt_str)

if __name__ == "__main__":
    main()