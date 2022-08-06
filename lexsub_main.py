#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context

# suggested imports
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

from collections import defaultdict

import numpy as np
import tensorflow

import gensim
import transformers

from typing import List

def tokenize(s):
    """
    a naive tokenizer that splits on punctuation and whitespaces.
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())
    return s.split()

def get_candidates(lemma, pos) -> List[str]:
    # get synsets
    synsets = wn.synsets(lemma, pos)
    # get name of each lemmas in each synset
    w = [l.name() for s in synsets for l in s.lemmas()]
    # return list of distinct lemmas w/o '_' separation char
    return list({x.replace('_', ' ') for x in w if x != lemma})

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context: Context) -> str:
    # get lemmas of all possible synonyms
    # (with duplication, for other senses)
    synsets = wn.synsets(context.lemma, context.pos)
    lemmas = [l for s in synsets for l in s.lemmas()]

    # count the occurances of each lemma in the corpus
    counts = defaultdict(int)
    for lemma in lemmas:
        counts[lemma.name()] += lemma.count()

    return max(counts).replace('_', ' ')

def wn_simple_lesk_predictor(context : Context) -> str:
    return None #replace for part 3


class Word2VecSubst(object):
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

    def predict_nearest(self,context : Context) -> str:
        return None # replace for part 4


class BertPredictor(object):
    def __init__(self):
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context : Context) -> str:
        return None # replace for part 5



if __name__ == "__main__":

    # At submission time, this program should run your best predictor (part 6).

    #W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    #predictor = Word2VecSubst(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
#         print(context)  # useful for debugging
        prediction = wn_frequency_predictor(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
