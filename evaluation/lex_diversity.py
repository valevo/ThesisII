# -*- coding: utf-8 -*-

from data.reader import wiki_from_pickles, corpora_from_pickles
from data.corpus import Sentences


from lexical_diversity import lex_div

import matplotlib.pyplot as plt
import seaborn as sns

def mtld_dists(subcorp_sets, names):
    for subcorp_set, name in zip(subcorp_sets, names):
        mtlds = [lex_div.mtld(list(subcorp.tokens())) for subcorp in subcorp_set]
        
        sns.distplot(mtlds, label=name)
        print(name)
        
    plt.xlabel("MTLD")
    plt.legend()
    plt.show()
    
    
def hdd_dists(subcorp_sets, names):
    for subcorp_set, name in zip(subcorp_sets, names):
        mtlds = [lex_div.hdd(list(subcorp.tokens())) for subcorp in subcorp_set]
        
        sns.distplot(mtlds, label=name)
        print(name)
        
    plt.xlim(0.5, 1.0)
    plt.xlabel("HD-D")
    plt.legend()
    plt.show()


def main(subcorp_sets, names):
    mtld_dists(subcorp_sets, names)
    hdd_dists(subcorp_sets, names)
    
    

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