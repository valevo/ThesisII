
        # -*- coding: utf-8 -*-

from data.reader import wiki_from_pickles, corpora_from_pickles
from data.corpus import Sentences


from lexical_diversity import lex_div

import matplotlib.pyplot as plt

if __name__ == "__main__":    
    n = 100000
    d = "results/ALS/"
    
    
    srf_samples = corpora_from_pickles(d + "SRF", names=["n", "h", "i"])
    srf_30 = [Sentences(c) for name_d, c in srf_samples if name_d["n"] == n and 
                                                  name_d["h"] == 30]
    
    tf_samples = corpora_from_pickles(d + "TF", names=["n", "f", "i"])
    tf_100 = [Sentences(c) for name_d, c in tf_samples if name_d["n"] == n and 
                                                  name_d["f"] == 100]
    
    uni_samples = corpora_from_pickles(d + "UNI", names=["n", "i"])
    uni = [Sentences(c) for name_d, c in uni_samples if name_d["n"] == n]
    
    print("loaded...")
    
    mtld_dist_srf = [lex_div.mtld(list(ss.tokens())) for ss in srf_30]
    print("SRF lex div...")
    mtld_dist_tf = [lex_div.mtld(list(ss.tokens())) for ss in tf_100]
    print("TF lex div...")
    mtld_dist_uni = [lex_div.mtld(list(ss.tokens())) for ss in uni]
    print("UNI lex div...")
    
    
    print(mtld_dist_srf)
    
    plt.hist(mtld_dist_srf, bins=7, color="blue", label="SRF 30")
    plt.hist(mtld_dist_tf, bins=7, color="red", label="TF 100")
    plt.hist(mtld_dist_uni, bins=7, color="yellow", label="UNI")
    
    plt.legend()
    plt.show()
    
    