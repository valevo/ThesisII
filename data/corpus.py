# -*- coding: utf-8 -*-

import numpy.random as rand

class Corpus:
    def __init__(self, corpus_list):
        self.elements = corpus_list

    def elements(self):
        yield from self.elements
        
    def sentences(self):
        pass

    def tokens(self):
        pass
        
    def types(self):
        return set(self.tokens())
    
    def __len__(self):
        return len(self.elements)

class Articles(Corpus):
    def sentences(self):
        for art in self.elements:
            yield from art
    
    def tokens(self):
        for art in self.elements:
            for sent in art:
                yield from sent
   
    @classmethod             
    def subsample(cls, corpus_list, n_tokens):
        def sample(articles, n_tokens):    
            m  = 0
            
            rand_inds = rand.choice(2, size=len(articles))
   
            for i, a in zip(rand_inds, articles):
                if i:
                    yield a
                    m += len([w for s in a for w in s])
                    
                    if m >= n_tokens:
                        break
                    
            if m < n_tokens:
                raise RuntimeError("not enough tokens sampled in `subsample_articles`!"
                                   f" ({m} < {n_tokens})")
                        
        subcorpus = sample(corpus_list, n_tokens)
        
        return cls(list(subcorpus))

class Sentences(Corpus):
    def sentences(self):
        yield from self.elements
    
    def tokens(self):
        for sent in self.elements:
            yield from sent
            
            
    @classmethod
    def subsample(cls, corpus_list, n_tokens):
        def sample(articles, n_tokens):
            m  = 0
    
            len_sents = len([s for a in articles for s in a])
            sent_iter = (s for a in articles for s in a)    
        
            rand_inds = rand.choice(2, size=len_sents)
    
            for i, s in zip(rand_inds, sent_iter):
                if i:
                    yield s
                    m += len(s)
        
                if m >= n_tokens:
                    break
        
            if m < n_tokens:
                raise RuntimeError("not enough tokens sampled in `subsample_sentences`!"
                                   f" ({m} < {n_tokens})")

        subcorpus = sample(corpus_list, n_tokens)
        return cls(list(subcorpus))

class Words(Corpus):    
    def sentences(self):
        raise NotImplementedError("Only tokens available in class Words!")
    
    def tokens(self):
        yield from self.elements
        
    @classmethod
    def subsample(cls, corpus_list, n_tokens):
        def sample(articles, n_tokens):
            m  = 0
    
            len_tokens = len([w for a in articles for s in a for w in s])
            word_iter = (w for a in articles for s in a for w in s)    
        
            rand_inds = rand.choice(2, size=len_tokens)
    
            for i, w in zip(rand_inds, word_iter):
                if i:
                    yield w
                    m += 1
        
                if m >= n_tokens:
                    break
        
            if m < n_tokens:
                raise RuntimeError("not enough tokens sampled in `subsample_tokens`!"
                                   f" ({m} < {n_tokens})")
                
                
        subcorpus = sample(corpus_list, n_tokens)
        return cls(list(subcorpus))
        