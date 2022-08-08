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
    return list({x.replace('_', ' ').lower() for x in w if x != lemma})

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

    def predict_nearest(self, context: Context) -> str:
        # cosine similarity w/ error handling
        def sim(x, y):
            try:
                return self.model.similarity(x, y)
            except KeyError:
                return 0

        # get list of possible synonyms
        syns = get_candidates(context.lemma, context.pos)
        # compute cosine similarity between target word
        sims = [sim(context.lemma, w) for w in syns]
        # return highest scoring word
        return syns[np.argmax(sims)]


class BertPredictor(object):
    def __init__(self):
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

    def predict(self, context: Context) -> str:
        # get list of possible synonyms
        syns = get_candidates(context.lemma, context.pos)

        # convert context to masked + encoded representation
        con = [w.lower() for w in context.left_context] \
               + ['[MASK]'] + [w.lower() for w in context.right_context]
        enc = self.tokenizer.encode(con, return_tensors='tf')
        pos = 1 + len(context.left_context) # encode adds [CLS] token to start

        # get BERT predictions for the target word
        pred = self.model.predict(enc, verbose=0)[0][0,pos]

        # select highest scoring word in syns
        syns_ids = self.tokenizer.encode(syns)
        syns_scores = pred[syns_ids[1:-1]] # don't grab [CLS] or [SEP]
        return syns[np.argmax(syns_scores)]


class W2VLesk(object):
    """
    Part 6, model 1.
    Find the best synset using the extended lesk algorithm,
    then choose the best word in the synset by Word2Vec cosine similarity
    (instead of frequency as used in part 4)

    Surprisingly, this does worse than either lesk or w2v...
    Perhaps the counts more accurately represent how likely a word is
    to appear in a certain context.
    """
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

    def compute_overlap(self, context_string, synset, bad_words):
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

    def sim(self, x, y):
        """ cosine similarity with error handling """
        try:
            return self.model.similarity(x, y)
        except KeyError:
            return 0

    def predict(self, context):
        """
        Find the best synset using the extended lesk algorithm,
        then choose the best word in the synset by Word2Vec cosine similarity
        (instead of frequency as used in part 4)

        Confusingly, this does worse than either lesk or w2v...
        """
        # ignore synsets where the target word is the only lemma
        synsets = [s for s in wn.synsets(context.lemma, context.pos)
                   if len(s.lemmas()) > 1]

        # full tokenized context string
        sw = stopwords.words('english')
        c = ' '.join(context.left_context + context.right_context)
        c = tokenize_and_remove(c, sw)

        # compute overlap scores
        # (ignore stop words and the target word)
        overlap = [self.compute_overlap(c, s, [context.lemma] + sw)
                   for s in synsets]

        # count number of times the target word appears in each sense
        freq = [l.count() for s in synsets for l in s.lemmas()
                if l.name() == context.lemma]

        # select best synset.
        # predominantly judge overlap, use freq as a tiebreaker
        scores = [1000*x[0] + x[1] for x in zip(overlap, freq)]
        best_syn = synsets[np.argmax(scores)]

        # return lemma in synset w/ highest cosine similarity to target word
        lemmas = [l.name() for l in best_syn.lemmas()
                  if l.name() != context.lemma]
        sims = [self.sim(context.lemma, l) for l in lemmas]
        return lemmas[np.argmax(sims)]


class BertWithWord2Vec(object):
    """
    Part 6, model 2.
    Same concept as the BertPredictor, but expand the candidate pool using Word2Vec.
    Gets better results than BertPredictor for me, but worse than w2v alone.
    """
    def __init__(self, w2vfile):
        # Word2Vec model, for computing model similarities
        self.w2v = gensim.models.KeyedVectors.load_word2vec_format(w2vfile, binary=True)
        # bert model, for predicting appropriate replacement words
        self.bert = transformers.TFDistilBertForMaskedLM \
                                .from_pretrained('distilbert-base-uncased')
        self.tokenizer = transformers.DistilBertTokenizer \
                                     .from_pretrained('distilbert-base-uncased')
        # vocab of the bert model, special tokens/chars removed
        sw = stopwords.words('english')
        self.vocab = [w for w in self.tokenizer.get_vocab() \
                      if len(w) > 2 and '[' not in w and '#' not in w and w not in sw]

    def sim(self, x, y):
        """ cosine similarity with error handling """
        try:
            return self.w2v.similarity(x, y)
        except KeyError:
            return 0

    def predict(self, context):
        # get list of possible synonyms
        syns = get_candidates(context.lemma, context.pos)

        # words in bert's vocab that are more similar to the target word
        # than the ones suggested by wordnet, according to w2v similarity
        # (use max(min_sim, 0.35) to avoid pulling in erroneous words)
        min_sim = max(self.sim(context.lemma, s) for s in syns)
        bert_syns = [wn.morphy(w, context.pos) for w in self.vocab
                     if wn.morphy(w, context.pos)
                     and self.sim(context.lemma, w) > max(min_sim, 0.35)
                     and wn.morphy(w, context.pos) != context.lemma]

        # expanded set of replacement candidates
        exp_syns = list(set(syns + bert_syns))
        if len(exp_syns) > 512:
            print('exp_syns len: ', len(exp_syns))
            print(context)
            print(exp_syns[0:300])

        # convert context to masked + encoded representation
        con = [w.lower() for w in context.left_context] \
               + ['[MASK]'] + [w.lower() for w in context.right_context]
        enc = self.tokenizer.encode(con, return_tensors='tf')
        pos = 1 + len(context.left_context) # encode adds [CLS] token

        # get BERT predictions for the target word
        pred = self.bert.predict(enc, verbose=0)[0][0,pos]

        # select highest scoring word in syns
        syns_ids = self.tokenizer.encode(exp_syns)
        syns_scores = pred[syns_ids[1:-1]] # don't grab [CLS] or [SEP]
        return exp_syns[np.argmax(syns_scores)]


if __name__ == "__main__":
    # At submission time, this program should run your best predictor (part 6).

#     nltk.download('stopwords')
    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'

    # models from p1-p5
#     predictor = wn_frequency_predictor
#     predictor = wn_simple_lesk_predictor
    predictor = Word2VecSubst(W2VMODEL_FILENAME).predict_nearest
#     predictor = BertPredictor().predict

    # custom models from part 6
#     predictor = W2VLesk(W2VMODEL_FILENAME).predict
#     predictor = BertWithWord2Vec(W2VMODEL_FILENAME).predict

    # useful for debugging
#     model = BertPredictor()
#     model = BertWithWord2Vec(W2VMODEL_FILENAME)
#     reader = read_lexsub_xml('lexsub_trial.xml')
#     context = next(reader)
#     while context.lemma != 'can':
#         context = next(reader)

    for context in read_lexsub_xml(sys.argv[1]):
#         print(context)  # useful for debugging
        prediction = predictor(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
