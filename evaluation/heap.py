# -*- coding: utf-8 -*-

from data.corpus import Sentences
from data.reader import wiki_from_pickles, corpora_from_pickles

from stats.stat_functions import compute_vocab_size

import numpy.random as rand

import matplotlib.pyplot as plt


def sent_subsample(sents, n_toks):
    n_sampled = 0
    
    used = set()
    
    while n_sampled < n_toks:
        cur_ind = rand.randint(len(sents))
        if cur_ind in used:
            continue
        sampled_sent = sents.elements[cur_ind]
        
        yield sampled_sent
        n_sampled += len(sampled_sent)
        used.add(cur_ind)
        
        
def heap(corp, rng):
    vocab_sizes = []
    for ntoks in rng:
        subsample = Sentences(sent_subsample(corp, ntoks))
        vocab_size = compute_vocab_size(subsample)
        vocab_sizes.append(vocab_size)
        
    return vocab_sizes

    

def heap_main(subcorp_sets, names):
    
    rng = list(range(10))
    
    voc_size_ls = []
    
    for subcorp_set, name in zip(subcorp_sets, names):
        for i, subcorp in enumerate(subcorp_set):
            voc_sizes = heap(subcorp, rng)
            voc_size_ls.append(voc_sizes, name, i)
            
    return voc_size_ls


if __name__ == "__main__":
    n = 1000000
    d = "results/ID/"
    
    ## LOAD CORPORA
    # SRFs    
    srf_samples = list(corpora_from_pickles(d + "SRF", names=["n", "h", "i"]))
    srf_10 = [Sentences(c) for name_d, c in srf_samples if name_d["n"] == n and 
                                                  name_d["h"] == 4]
    srf_20 = [Sentences(c) for name_d, c in srf_samples if name_d["n"] == n and 
                                                  name_d["h"] == 32]
    srf_30 = [Sentences(c) for name_d, c in srf_samples if name_d["n"] == n and 
                                                  name_d["h"] == 81]
    #TFs
    tf_samples = list(corpora_from_pickles(d + "TF", names=["n", "f", "i"]))
    tf_50 = [Sentences(c) for name_d, c in tf_samples if name_d["n"] == n and 
                                                  name_d["f"] == 2]  
    tf_100 = [Sentences(c) for name_d, c in tf_samples if name_d["n"] == n and 
                                                  name_d["f"] == 18]    
    #UNIs
    uni_samples = corpora_from_pickles(d + "UNI", names=["n", "i"])
    uni = [Sentences(c) for name_d, c in uni_samples if name_d["n"] == n]
    
    
    
    ntoks = list(range(n//1000, n, n//50))
    
    for subcorp_set, name, colour in zip([srf_10, srf_20, srf_30, tf_50, tf_100, uni], 
                                 ["SRF10", "SRF20", "SRF30", "TF50", "TF100", "UNI"],
                                 ["green", "blue", "brown", "yellow", "red", "purple"]):
    
        for i, subcorp in enumerate(subcorp_set):
        
#            rand_ind = rand.randint(len(subcorp_set))
            print(i)
#            
#            subcorp = subcorp_set[rand_ind]
#        
            vocab_sizes = []
            for n in ntoks:
                subsample = Sentences(sent_subsample(subcorp, n))

                vcb_len = compute_vocab_size(subsample)
#                print(vcb_len, end=" ", flush=True)

                vocab_sizes.append(vcb_len)

            print("\n")
            plt.plot(ntoks, vocab_sizes, '--', color=colour, 
                     label=(name if i == 1 else None))
        
    plt.legend()
    plt.show()