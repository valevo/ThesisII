# -*- coding: utf-8 -*-

from data.reader import wiki_from_pickles, corpus_to_pickle
from filtering.typicality_parallelised import setup_filtering,\
                filter_typicality_incremental


import multiprocessing as mp
from multiprocessing import Array
from ctypes import c_wchar_p
import os
import numpy.random as rand


import argparse

def parse_args():
    p = argparse.ArgumentParser()
    
    p.add_argument("--lang", type=str)
    p.add_argument("--n_tokens", type=int)
    p.add_argument("--factor", type=float,
                   help="The factor to multiply epsilon with, determines"
                   "the degree of atypicality.")
    
    args = p.parse_args()
    return args.lang, args.n_tokens, args.factor



sep = "■"
def sents_to_mp_array(sents):
    sents_joined = [sep.join(s) for s in sents]
    return Array(c_wchar_p, sents_joined)

if __name__ == "__main__":
    lang, n, factor = parse_args()
    big_n = lambda wiki: len([w for a in wiki for s in a for w in s])*.49
    setup_m = 50
    m = 10
    
    wiki = list(wiki_from_pickles("data/"+lang+"_pkl"))
    sents = [s for a in wiki for s in a]
    
    
    big_corpus_size = big_n(wiki)
    print("n_tokens wiki: ", len([w for a in wiki for s in a for w in s]))
    print("n_tokens for big corpus: ", big_corpus_size)

    zipf_model, rank_dict, auto_typicality, epsilon = setup_filtering(wiki, 
                                                                      big_corpus_size, 
                                                                      n, 
                                                                      setup_m)
    print("\nModel and Epsilon established")
    print(auto_typicality, epsilon)
    print()
    
    
    mp_array = sents_to_mp_array((s for a in wiki for s in a))
    
    def filter_worker(i):
        print("started ", i)
        cur_seed = int.from_bytes(os.urandom(4), byteorder='little')
        rand.seed(cur_seed)
        filtered = list(filter_typicality_incremental(mp_array, zipf_model, 
                        rank_dict, auto_typicality, factor*epsilon, n))
        print("filtered ", i)
    
        
        name = "_".join((str(n), str(factor), str(i)))
        corpus_to_pickle(filtered, "results/" + lang + "/TF", name)
        
    
    i_ls = list(range(m))
    
    with mp.Pool(6) as p:
        p.map(filter_worker, i_ls)