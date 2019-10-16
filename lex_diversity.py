# -*- coding: utf-8 -*-

from data.reader import wiki_from_pickles, corpora_from_pickles
from data.corpus import Sentences


from lexical_diversity import lex_div

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
    

    for subcorp_set, name, colour in zip([srf_10, srf_20, srf_30, tf_50, tf_100, uni], 
                                 ["SRF10", "SRF20", "SRF30", "TF50", "TF100", "UNI"],
                                 ["green", "blue", "brown", "yellow", "red", "purple"]):
        
        mtlds = [lex_div.mtld(list(subcorp.tokens())) for subcorp in subcorp_set]
        print(name)
        
        plt.hist(mtlds, bins=5, color=colour, label=name)
    
    plt.title("MTLD")    
    plt.legend()
    plt.show()
    
    
    for subcorp_set, name, colour in zip([srf_10, srf_20, srf_30, tf_50, tf_100, uni], 
                                 ["SRF10", "SRF20", "SRF30", "TF50", "TF100", "UNI"],
                                 ["green", "blue", "brown", "yellow", "red", "purple"]):
        
        mtlds = [lex_div.hdd(list(subcorp.tokens())) for subcorp in subcorp_set]
        print(name)
        
        plt.hist(mtlds, bins=5, color=colour, label=name)
    
    plt.title("HD-D")    
    plt.legend()
    plt.show()


    
    
    
    
    
    
    
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
    
    plt.title("MTLD Distribution")
    plt.legend()
    plt.show()
    
    
    hdd_dist_srf = [lex_div.hdd(list(ss.tokens())) for ss in srf_30]
    print("SRF lex div...")
    hdd_dist_tf = [lex_div.hdd(list(ss.tokens())) for ss in tf_100]
    print("TF lex div...")
    hdd_dist_uni = [lex_div.hdd(list(ss.tokens())) for ss in uni]
    print("UNI lex div...")
    
    
    print(hdd_dist_srf)
    
    plt.hist(hdd_dist_srf, bins=7, color="blue", label="SRF 30")
    plt.hist(hdd_dist_tf, bins=7, color="red", label="TF 100")
    plt.hist(hdd_dist_uni, bins=7, color="yellow", label="UNI")
    
    plt.title("HD-D Distribution")
    plt.legend()
    plt.show()
    
    
    