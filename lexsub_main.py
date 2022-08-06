#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context

# suggested imports
import string
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

# casey oneills imports
from collections import defaultdict
import nltk

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

def tokenize_and_remove(s, r):
    """
    the same naive tokenizer, that also deletes words in the set r.
    """
    # tokenize
    s = ''.join(' ' if x in string.punctuation else x for x in s.lower())
    # remove
    return [w for w in s.split() if w not in r]

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
    # count the occurances of each lemma in the corpus
    counts = defaultdict(int)
    for synset in wn.synsets(context.lemma, context.pos):
        # this will hit same lemma multiple times
        # if it belongs to multiple synsets
        for lemma in synset.lemmas():
            if lemma.name() != context.lemma:
                counts[lemma.name()] += lemma.count()

    # return lemma w/ highest occurance count
    return max(counts, key=counts.get).replace('_', ' ')

def wn_simple_lesk_predictor(context : Context) -> str:
    def compute_overlap(context_string, synset, bad_words):
        """
        compute overlap between context string and synset.
        removes words in bad_words from definition lists.
        """
        # definition of synset
        d = tokenize_and_remove(synset.definition(), bad_words)

        # extend w/ examples of synset
        for ex in synset.examples():
            d += tokenize_and_remove(ex, bad_words)

        # extend w/ hypernyms
        for h in synset.hypernyms():
            d += tokenize_and_remove(h.definition(), bad_words)
            for ex in h.examples():
                d += tokenize_and_remove(ex, bad_words)

        # count overlap between def and context
        # this approach double counts, i.e.
        #     [a, b, b],    [b] -> 2
        #     [a, b, b], [b, b] -> 4
        return sum(1 for x in context_string for y in d if x == y)

    # ignore synsets where the target word is the only lemma
    synsets = [s for s in wn.synsets(context.lemma, context.pos)
               if len(s.lemmas()) > 1]

    # full tokenized context string
    sw = stopwords.words('english')
    c = ' '.join(context.left_context + context.right_context)
    c = tokenize_and_remove(c, sw)

    # compute overlap scores
    # (ignore stop words and the target word)
    overlap = [compute_overlap(c, s, [context.lemma] + sw)
               for s in synsets]

    # count number of times the target word appears in each sense
    freq = [l.count() for s in synsets for l in s.lemmas()
            if l.name() == context.lemma]

    # select best synset.
    # predominantly judge overlap, use freq as a tiebreaker
    scores = [1000*x[0] + x[1] for x in zip(overlap, freq)]
    best_syn = synsets[np.argmax(scores)]

    # return most frequent synonym from the best synset
    syns = [l for l in best_syn.lemmas() if l.name() != context.lemma]
    return max(syns, key=lambda l: l.count()).name()


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

#     nltk.download('stopwords')
#     reader = read_lexsub_xml('lexsub_trial.xml')
#     context = next(reader)
    for context in read_lexsub_xml(sys.argv[1]):
#         print(context)  # useful for debugging
        prediction = wn_simple_lesk_predictor(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
