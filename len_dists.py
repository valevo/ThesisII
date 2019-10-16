# -*- coding: utf-8 -*-

from data.reader import corpora_from_pickles
from data.corpus import Sentences


import matplotlib.pyplot as plt


if __name__ == "__main__":
    n = 100000
    d = "results/ALS/"
    
    ## LOAD CORPORA
    # SRFs    
    srf_samples = list(corpora_from_pickles(d + "SRF", names=["n", "h", "i"]))
    srf_10 = [Sentences(c) for name_d, c in srf_samples if name_d["n"] == n and 
                                                  name_d["h"] == 10]
    srf_20 = [Sentences(c) for name_d, c in srf_samples if name_d["n"] == n and 
                                                  name_d["h"] == 20]
    srf_30 = [Sentences(c) for name_d, c in srf_samples if name_d["n"] == n and 
                                                  name_d["h"] == 30]
    #TFs
    tf_samples = list(corpora_from_pickles(d + "TF", names=["n", "f", "i"]))
    tf_50 = [Sentences(c) for name_d, c in tf_samples if name_d["n"] == n and 
                                                  name_d["f"] == 50]  
    tf_100 = [Sentences(c) for name_d, c in tf_samples if name_d["n"] == n and 
                                                  name_d["f"] == 100]    
    #UNIs
    uni_samples = corpora_from_pickles(d + "UNI", names=["n", "i"])
    uni = [Sentences(c) for name_d, c in uni_samples if name_d["n"] == n]
    
    
    # sent len dists
    for subcorp_set, name, colour in zip([srf_10, srf_20, srf_30, tf_50, tf_100, uni], 
                                 ["SRF10", "SRF20", "SRF30", "TF50", "TF100", "UNI"],
                                 ["green", "blue", "brown", "yellow", "red", "purple"]):
        
        
        all_lens = []
        for i, subcorp in enumerate(subcorp_set):
            lens = list(map(len, subcorp.sentences()))
            all_lens.extend(lens)
        
        i = 0
        plt.hist(all_lens, bins=100, color=colour, histtype="step",
                     label=(name if i == 0 else None))
        
        print(name)
    
    plt.title("Sentence Lengths")
    plt.legend()
    plt.show()
            
    
    
    # word len dists
    for subcorp_set, name, colour in zip([srf_10, srf_20, srf_30, tf_50, tf_100, uni], 
                                 ["SRF10", "SRF20", "SRF30", "TF50", "TF100", "UNI"],
                                 ["green", "blue", "brown", "yellow", "red", "purple"]):
        
        
        all_lens = []
        for i, subcorp in enumerate(subcorp_set):
            lens = [len(w) for s in subcorp.sentences() for w in s]
            all_lens.extend(lens)
        
        i = 0
        plt.hist(all_lens, bins=100, color=colour, histtype="step",
                     label=(name if i == 0 else None))
        
        print(name)
    
    plt.title("Word Lengths")
    plt.legend()
    plt.show()
            
            