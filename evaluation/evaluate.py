import pandas as pd
from pathlib import Path
import os
import numpy as np

import click
import math
import torch
import torch.nn as nn

import argparse
import json
import os
import operator
import logging
import random
import pickle 

from functools import partial
from collections import Counter
from scipy import stats

import string
logger = logging.getLogger(__name__)

def perplexity(generations, write_file=None):
    PATH = "../adaptive-softmax-pytorch/text8_lm.pt"

    def generate_square_subsequent_mask(sz: int):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

    model = torch.load(PATH)
    model.eval()

    ntokens = model.encoder.num_embeddings
    bptt = 256

    # src_mask = generate_square_subsequent_mask(bptt).cuda()#to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction="none")

    wordmap = {char: id for id, char in enumerate(string.ascii_lowercase)}
    wordmap[" "] = 26

    for i, gen in enumerate(generations):
        if len(gen) != 256:
            print(i, len(gen), gen)
            input()

    data = []
    for char in "".join(generations):
        try:
            data.append(wordmap[char])
        except KeyError:
            data.append(26)

    data = np.array(data)
    data = data.reshape(-1, 256)

    def energy(batch):
        with torch.no_grad():
            seq_len = batch.size(0)
            src_mask = generate_square_subsequent_mask(bptt).cuda()#to(device)
            src_mask = src_mask[:seq_len, :seq_len]
            output = model(batch, src_mask)[:-1, :]
            targets = batch[1:, :].view(-1)

            # print(targets.size())
            # print(output.size())
            nllloss = criterion(output.view(-1, ntokens), targets)
            # print(nllloss.size())
            nllloss = nllloss.view(seq_len-1, -1).mean(dim=0).tolist()
            # print(nllloss)
            # input()
            return nllloss

    batch_size = 128
    ppls = []
    for i in range(0, len(data), batch_size):
        batch = data[i:min(len(data),i+batch_size)]
        loss = energy(torch.from_numpy(batch).cuda().transpose(1, 0).contiguous())
        ppls += loss
        # for j in range(len(batch)):
            # ppl = torch.exp(loss)
            # ppls.append(ppl[j].item())
    
    return np.nanmean(ppls)


def fluency_classify(generations_df, output_file, batch_size=32):
    from fairseq.models.roberta import RobertaModel
    from fairseq.data.data_utils import collate_tokens

    model = RobertaModel.from_pretrained(
            '/projects/tir5/users/sachink/embed-style-transfer/evaluation_models/cola_classifier_fluency/',
            checkpoint_file='checkpoint_best.pt',
            data_name_or_path='./cola-bin'
        )
    model.cuda()

    def label_fn(label):
        return model.task.label_dictionary.string(
            [label + model.task.target_dictionary.nspecial]
        )
    
    def predict_batch(batch):
        batch = collate_tokens([model.task.source_dictionary.encode_line("<s> " + sd + " </s>", append_eos=False) for sd in batch], pad_idx=1)
        batch = batch[:, :512]

        with torch.no_grad():
            predictions = model.predict('sentence_classification_head', batch.long())
            # prediction_probs = [torch.exp(x).max(axis=0)[0].item() for x in predictions]
            prediction_labels = [label_fn(x.argmax(axis=0).item()) for x in predictions]
        
        return prediction_labels
            
    batch = []
    all_prediction_labels = []
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating CoLA fluency'):
        prompt = row.prompt['text']
        generations = [gen['text'] for gen in row['generations']]
        for j, gen in enumerate(generations):
            batch.append(model.bpe.encode(f'{prompt}{gen}'))
            if len(batch) == batch_size:
                prediction_labels = predict_batch(batch)
                all_prediction_labels += prediction_labels
                batch = []
        
        if len(batch) != 0:
            prediction_labels = predict_batch(batch)
            all_prediction_labels += prediction_labels
            batch = []
    
    with open(output_file, "w") as fout:
        fout.write("\n".join(all_prediction_labels))

    accuracy = np.array(all_prediction_labels) == "acceptable"
    accuracy = np.nanmean(accuracy.astype("float32"))
    return accuracy


def distinctness(generations_df):
    dist1, dist2, dist3 = [], [], []
    # calculate dist1, dist2, dist3 across generations for every prompt
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating dist-n'):
        generations = row['string']
        unigrams, bigrams, trigrams = set(), set(), set()
        total_words = 0
        for gen in generations:
            o = gen.split(' ')
            # o = [str(tok) for tok in gen]
            total_words += len(o)
            unigrams.update(o)
            for i in range(len(o) - 1):
                bigrams.add(o[i] + '_' + o[i+1])
            for i in range(len(o) - 2):
                trigrams.add(o[i] + '_' + o[i+1] + '_' + o[i+2])
        dist1.append(len(unigrams) / total_words)
        dist2.append(len(bigrams) / total_words)
        dist3.append(len(trigrams) / total_words)
    
    # take the mean across prompts
    return np.nanmean(dist1), np.nanmean(dist2), np.nanmean(dist3)

def self_bleu(generations_df, n_sample=1000):

    # import spacy
    random.seed(0)
    # nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])
    # nlp.add_pipe(nlp.create_pipe('sentencizer'))

    smoothing_function = SmoothingFunction().method1
    all_sentences = []
    for i, row in generations_df.iterrows():
        # gens = row['tokens']
        gens = [[str(token) for token in tokens] for tokens in row['tokens']]# for gen in row['generations']] {'prompt':"", tokens: [[1,2,3], [3,4,5], [5,6,7], ....]}
        all_sentences += gens
    
    pool = Pool(processes=os.cpu_count())
    bleu_scores = []
    for n_gram in range(1, 6):

        if n_gram == 1:
            weights = (1.0, 0, 0, 0)
        elif n_gram == 2:
            weights = (0.5, 0.5, 0, 0)
        elif n_gram == 3:
            weights = (1.0 / 3, 1.0 / 3, 1.0 / 3, 0)
        elif n_gram == 4:
            weights = (0.25, 0.25, 0.25, 0.25)
        elif n_gram == 5:
            weights = (0.2, 0.2, 0.2, 0.2, 0.2)
        else:
            raise ValueError
        bleu_scores.append(
            list(tqdm(
                pool.imap_unordered(
                    partial(bleu_i, weights, all_sentences, smoothing_function),
                    random.sample(range(len(all_sentences)), min(n_sample, len(all_sentences)))),
                total=min(n_sample, len(all_sentences)),
                smoothing=0.0,
                desc=f"bleu-{n_gram}")))
        # print(f"\n\nbleu-{n_gram} = {sum(bleu_scores[n_gram - 1]) / n_sample}")
    
    pool.close()
    pool.join()

    bleus = []
    for n_gram in range(5):
        bleus.append(sum(bleu_scores[n_gram]) / n_sample)
        # print(f"bleu-{n_gram + 1} = {sum(bleu_scores[n_gram]) / n_sample}")
    
    return bleus


def self_bleu2(generations_df, n_sample=100):

    # import spacy
    random.seed(0)
    smoothing_function = SmoothingFunction().method1
    # nlp = spacy.load('en', disable=['parser', 'tagger', 'ner'])
    # nlp.add_pipe(nlp.create_pipe('sentencizer'))
    all_bleus = [[] for _ in range(3)]
    for i, row in generations_df.iterrows():
        # all_sentences = []
        all_sentences = row['tokens']# for gen in row['generations']]
        # all_sentences += gens
        
        pool = Pool(processes=os.cpu_count())
        bleu_scores = []
        for i in range(3):
            n_gram = i+3
            if n_gram == 1:
                weights = (1.0, 0, 0, 0)
            elif n_gram == 2:
                weights = (0.5, 0.5, 0, 0)
            elif n_gram == 3:
                weights = (1.0 / 3, 1.0 / 3, 1.0 / 3, 0)
            elif n_gram == 4:
                weights = (0.25, 0.25, 0.25, 0.25)
            elif n_gram == 5:
                weights = (0.2, 0.2, 0.2, 0.2, 0.2)
            else:
                raise ValueError
            bleu_scores.append(
                list(tqdm(
                    pool.imap_unordered(
                        partial(bleu_i, weights, all_sentences, smoothing_function),
                        random.sample(range(len(all_sentences)), min(n_sample, len(all_sentences)))),
                    total=min(n_sample, len(all_sentences)),
                    smoothing=0.0,
                    desc=f"bleu-{n_gram}")))
            # print(f"\n\nbleu-{n_gram} = {sum(bleu_scores[n_gram - 1]) / n_sample}")
        
        pool.close()
        pool.join()

        for i in range(3):
            all_bleus[i].append(sum(bleu_scores[i]) / n_sample)
            # print(f"bleu-{n_gram + 1} = {sum(bleu_scores[n_gram]) / n_sample}")
    all_bleus = [np.nanmean(bleu) for bleu in all_bleus]
    return all_bleus


def repetition(generations_df, tokenizer, numbers_only=True, rep_file=None):
    SEP = tokenizer.encode(tokenizer.bos_token)[0]

    objs = []
    max_n = 90

    n_repeated_examples = 0
    total_examples = 0

    if rep_file is not None:
        fout = open(rep_file, "w")
    for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating repetitions'):
        generations = row['tokens'] #for gen in row['generations']]
        for gen in generations:
            total_examples += 1
            if gen[-1] == SEP:
                gen.pop(-1)
            rev_gen = list(reversed(gen))
            last_n_repeats = [0] * max_n

            for n in range(1, max_n + 1):
                n_repeat = 1
                while len(rev_gen[n*n_repeat:n*(n_repeat+1)]) == n and \
                        rev_gen[n*n_repeat:n*(n_repeat+1)] == rev_gen[:n]:
                    n_repeat += 1
                last_n_repeats[n - 1] = n_repeat
            max_repeated_n = max(range(max_n), key=lambda x: last_n_repeats[x])

            if last_n_repeats[max_repeated_n] > 1 and (max_repeated_n+1 >= 3 or last_n_repeats[max_repeated_n] > 50):
                repetition = {
                    'repeated_phrase': list(reversed(rev_gen[:max_repeated_n + 1])),
                    'repeated_times': last_n_repeats[max_repeated_n],
                    'repeated_phrase_length': max_repeated_n + 1,
                }
                n_repeated_examples += 1
            else:
                repetition = {}
            
            if rep_file is not None:
                json.dump(repetition, fout)
                fout.write("\n")
    
    if rep_file is not None:
        fout.close()

    return n_repeated_examples*1.0/total_examples


@click.command()
@click.option('--generations_file', required=True, type=str, help='a text file with generations and attribute scores')
@click.option('--output_file', required=True, type=str, help='filename to write the results to')
@click.option('--metrics', required=True, type=str, help='which metrics to compute, write comma separeted, ppl-mid,ppl-big,cola,self-bleu,zipf,repetition,dist-n')
@click.option('--extra', required=False, type=str, help='extra params')
def main(generations_file, output_file, metrics, extra):
    assert os.path.exists(generations_file)
    output_dir = Path(os.path.dirname(generations_file))
    # generations_df = pd.read_json(generations_file, lines=True)
    generations = [line.rstrip("\n") for line in open(generations_file)]
    
    metricset = set(metrics.strip().lower().split(","))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### calculate quality metrics

    # Fluency
    fo = open(output_dir / output_file, 'w') #creating the file
    fo.close()
    
    # print(metrics)
    if "ppl" in metrics:
        print(f'computing energy model ppl')
        ppl = perplexity(generations, write_file=None)#output_dir / (output_file+".ppl-"+eval_modelname))
        print(ppl)
        # # write output results
        # with open(output_dir / output_file, 'a') as fo:
        #     fo.write(f'{eval_modelname} perplexity, {eval_modelname} total perplexity = {ppl}, {total_ppl}\n')
        #     print(f'{eval_modelname} perplexity, {eval_modelname} total perplexity = {ppl}, {total_ppl}\n')
    
    #cola
    if "cola" in metricset:
        print("computing fluency (cola)")
        cola_accuracy = fluency_classify(generations_df, output_file=output_dir / (output_file+".cola"))
        
        # write output results
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'cola acceptability accuracy = {cola_accuracy}\n')
            print(cola_accuracy)

    ### calculate diversity
    # dist-n
    if "dist-n" in metricset:
        dist1, dist2, dist3 = distinctness(generations_df)
        
        # write output results
        with open(output_dir / output_file, 'a') as fo:
            for i, dist_n in enumerate([dist1, dist2, dist3]):
                fo.write(f'dist-{i+1} = {dist_n}\n')
                print(f'dist-{i+1} = {dist_n}')

    # self-bleu
    if "self-bleu" in metricset:
        bleus = self_bleu(generations_df)
        with open(output_dir / output_file, 'a') as fo:
            for i, bleu in enumerate(bleus):
                fo.write(f'bleu-{i+1} = {bleu}\n')
                print(f'bleu-{i+1} = {bleu}')
    
    # self-bleu
    if "self-bleu2" in metricset:
        bleus = self_bleu2(generations_df)
        with open(output_dir / output_file, 'a') as fo:
            for i, bleu in enumerate(bleus):
                fo.write(f'bleu-{i+3} = {bleu}\n')
                print(f'bleu-{i+3} = {bleu}')

    # repetition
    if "repetition" in metricset:
        eval_tokenizer = AutoTokenizer.from_pretrained('gpt2-large')
        rep_rate = repetition(generations_df, eval_tokenizer, rep_file=output_dir / (output_file+".repetitions"))
        with open(output_dir / output_file, 'a') as fo:
            fo.write(f'repetition_rate: {rep_rate}\n')
            print(f'repetition_rate: {rep_rate}')

if __name__ == '__main__':
    main()