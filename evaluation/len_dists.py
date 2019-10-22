# -*- coding: utf-8 -*-

from data.reader import corpora_from_pickles
from data.corpus import Sentences

from jackknife.plotting import colour_palette

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import numpy.random as rand

sns.set_palette(colour_palette)


def sent_len_dists(subcorp_sets, names):
    all_xs = []
    all_ys = []
    means_stddevs = dict()
    for subcorp_set, name in zip(subcorp_sets, names):
        
        i = rand.randint(len(subcorp_set))
        
        all_lens = [len(s) for s in subcorp_set[i].sentences() if len(s) < 40
                    and len(s) > 0]
        all_xs.extend(all_lens)
        
        lbls = [name]*len(all_lens)
        all_ys.extend(lbls)
        
        means_stddevs[name] = (np.mean(all_lens), np.var(all_lens)**.5)
    
    print(means_stddevs)
    sns.violinplot(x=all_xs, y=all_ys, cut=0)
    plt.xlabel("Sentence Length")
    plt.show()    
    return means_stddevs
    

def word_len_dists(subcorp_sets, names):
    all_xs = []
    all_ys = []
    means_stddevs = dict()
    for subcorp_set, name in zip(subcorp_sets, names):
        
        i = rand.randint(len(subcorp_set))
        
        all_lens = [len(w) for w in subcorp_set[i].tokens() 
            if len(w) < 20 and len(w) > 0]
        all_xs.extend(all_lens)
        
        lbls = [name]*len(all_lens)
        all_ys.extend(lbls)
        
        means_stddevs[name] = (np.mean(all_lens), np.var(all_lens)**.5)
    
    print(means_stddevs)
    sns.violinplot(x=all_xs, y=all_ys, cut=0)
    plt.xlabel("Word Length")
    plt.show()    
    return means_stddevs


def main(subcorp_sets, names):
    sent_mean_dict = sent_len_dists(subcorp_sets, names)
    word_mean_dict = word_len_dists(subcorp_sets, names)
    


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
    
    
    subcorp_sets = [srf_10, srf_20, srf_30, tf_50, tf_100, uni]
    names = ["SRF10", "SRF20", "SRF30", "TF50", "TF100", "UNI"]
    
    
    main(subcorp_sets, names)
    
