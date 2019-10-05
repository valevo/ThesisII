# -*- coding: utf-8 -*-
import os
import pickle
        
def wiki_from_pickles(pkls_dir, drop_titles=True, n_tokens=int(50e6)):
    pkl_files = os.listdir(pkls_dir)
    
    n_loaded = 0
    
    for f in pkl_files:
        with open(pkls_dir + "/" + f, "rb") as handle:
            wiki = pickle.load(handle)
            for title, a in wiki:
                n_loaded += len([w for s in a for w in s])
                yield a if drop_titles else (title, a)
                
                if n_loaded >= n_tokens:
                    break
                
                
def corpus_to_pickle(corpus, pkl_dir, pkl_name):
    with open(pkl_dir + "/" + pkl_name + ".pkl", "wb") as handle:
        pickle.dump(corpus, handle)
        
        
