# -*- coding: utf-8 -*-

from data.reader import corpora_from_pickles
from data.corpus import Sentences

from evaluation.lex_diversity import main as lex_main
from evaluation.len_dists import main as len_main
from evaluation.heap import heap_main



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
    
    
    ## LENGTH DISTS
#    len_main(subcorp_sets, names)
    
    
    ## LEX DIV MEASURES
#    lex_mhain(subcorp_sets, names)
    
    
    ## HEAP 
    heap_main(subcorp_sets, names, rng=range(1000, 100000, 5000))