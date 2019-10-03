# -*- coding: utf-8 -*-
import os
import pickle
        
def wiki_from_pickles(pkls_dir, drop_titles=True):
    pkl_files = os.listdir(pkls_dir)
    
    for f in pkl_files:
        with open(pkls_dir + "/" + f, "rb") as handle:
            wiki = pickle.load(handle)
            if drop_titles:
                for title, a in wiki:
                    yield a
            else:
                yield from wiki
                
                
def corpus_to_pickle(corpus, pkl_dir, pkl_name):
    with open(pkl_dir + "/" + pkl_name + ".pkl", "wb") as handle:
        pickle.dump(corpus, handle)