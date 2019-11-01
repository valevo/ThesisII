# -*- coding: utf-8 -*-
from collections import Counter
import numpy as np

def compute_freqs(corpus):
    type_counts = Counter(corpus.tokens())
    return type_counts

def compute_normalised_freqs(corpus):
    type_counts = Counter(corpus.tokens())
    n = sum(type_counts.values())
    return {w: f/n for w, f in type_counts.items()}


def compute_ranks(corpus):
    freqs = compute_freqs(corpus)
    return {w: r for r, (w, c) in enumerate(freqs.most_common(), 1)}


def merge_to_joint(ranks, freqs):
    common_types = ranks.keys() & freqs.keys()
    return {w : (ranks[w], freqs[w]) for w in common_types}


def pool_stats(stat_ls, join_func=set.union):
    common_types = join_func(
            *[set(stat_d.keys()) for stat_d in stat_ls]
            )
    
    return {w: [stat_d[w] for stat_d in stat_ls if w in stat_d] 
            for w in common_types}    

def reduce_pooled(pooled_stats, reduce_func=np.mean):
    return {w: reduce_func(stats) for w, stats in pooled_stats.items()}

def compute_vocab_size(corpus):
    return len(corpus.types())

def compute_hapax_size(corpus, k):
    freqs = compute_freqs(corpus)
    return len([w for w, f in freqs if f == k])


def pool_heap(stat_ls):
    pass

