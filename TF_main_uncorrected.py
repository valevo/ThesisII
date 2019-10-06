# -*- coding: utf-8 -*-

from data.reader import wiki_from_pickles, corpus_to_pickle
from stats.entropy import typicality
from filtering.typicality_uncorrected import setup_filtering, filter_typicality_incremental

import argparse

def parse_args():
    p = argparse.ArgumentParser()
    
    p.add_argument("--lang", type=str)
    p.add_argument("--n_tokens", type=int)
    p.add_argument("--factor", type=float,
                   help="The factor to multiply epsilon with; determines"
                   "the degree of atypicality.")
    
    args = p.parse_args()
    return args.lang, args.n_tokens, args.factor

if __name__ == "__main__":
    lang, n, factor = parse_args()
    big_n = lambda wiki: len([w for a in wiki for s in a for w in s])*.49
    setup_m = 50
    m = 10
    
    wiki = list(wiki_from_pickles("data/"+lang+"_pkl"))
    sents = [s for a in wiki for s in a]

    zipf_model, rank_dict, mean_typ, std_typ, auto_typ = setup_filtering(wiki, 
                                                                         big_n(wiki), 
                                                                         n, 
                                                                         setup_m)
    print("\nModel and Epsilon established")
    print(auto_typ, mean_typ, std_typ)
    
    
    epsilon = mean_typ - factor*std_typ
    print("epsilon: ", epsilon)
    print()
    
    for m_i in range(m):
        print("started ", m_i)        
        filtered = list(filter_typicality_incremental(sents, zipf_model, 
                        rank_dict, epsilon, n))
        print("filtered ", m_i)

        
        name = "_".join((str(n), str(factor), str(m_i)))
        corpus_to_pickle(filtered, "results/" + lang + "/TF", name)