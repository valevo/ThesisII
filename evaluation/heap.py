# -*- coding: utf-8 -*-

from data.corpus import Sentences
from data.reader import wiki_from_pickles, corpora_from_pickles

from stats.stat_functions import compute_vocab_size
from stats.mle import Heap

from jackknife.plotting import hexbin_plot, colour_palette

import numpy as np
import numpy.random as rand

import matplotlib.pyplot as plt
import seaborn as sns

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



def vocab_growth_compare(vocab_size_sets, names, rng):
    for i, (vocab_size_set, name) in enumerate(zip(vocab_size_sets, names)):
        rand_i = rand.randint(len(vocab_size_set))
        print(rand_i)
        v_ns = vocab_size_set[rand_i]
        heap = Heap(vocab_sizes, rng)
        heap_fit = heap.fit(start_params=np.asarray([1.0, 1.0]), 
                                    method="powell", full_output=True)    
        heap.register_fit(heap_fit)
        
        hexbin_plot(rng, v_ns, xlbl="$n$", ylbl="$V(n)$",
                    log=False, ignore_zeros=False, label=name,
                    color=colour_palette[i], edgecolors=colour_palette[i],
                    linewidths=1.0, cbar=(True if i==0 else False))
    plt.legend(loc="upper left")
#    plt.savefig(save_dir + "vocab_growth_" + 
#                str(min(rng)) + "_" + str(max(rng)) + "_" + str(len(rng)) + ".png",
#                dpi=300)
    plt.show()
#    plt.close()    

def heap_param_compare(vocab_size_sets, names, rng):
    
    for i, (vocab_size_set, name) in enumerate(zip(vocab_size_sets, names)):
        betas = []
        for vocab_sizes in vocab_size_set:
            heap = Heap(vocab_sizes, rng)
            heap_fit = heap.fit(start_params=np.asarray([1.0, 1.0]), 
                                    method="powell", full_output=True)    
            heap.register_fit(heap_fit)
            betas.append(heap.optim_params[1])
            
        sns.distplot(betas, label=name)
    plt.xlabel("$\tau$")
    plt.legend()
    plt.show()
    
    
def heap_main(subcorp_sets, names, rng):    
    vocab_sizes_all = []
    
    for subcorp_set, name in zip(subcorp_sets, names):
        cur_vocab_sizes = []
        for i, subcorp in enumerate(subcorp_set):
            voc_sizes = heap(subcorp, rng)
            cur_vocab_sizes.append(voc_sizes)
        vocab_sizes_all.append(cur_vocab_sizes)
        
    vocab_growth_compare(vocab_sizes_all, names, rng)
    
    heap_param_compare(vocab_sizes_all, names, rng)
    
    
    

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
    
    
    
    ntoks = list(range(n//1000, n, n//5))
    
    for subcorp_set, name, colour in zip([srf_10, srf_20, srf_30, tf_50, tf_100, uni], 
                                 ["SRF10", "SRF20", "SRF30", "TF50", "TF100", "UNI"],
                                 ["green", "blue", "brown", "yellow", "red", "purple"]):
        print()
        print(name)
        
        for i, subcorp in enumerate(subcorp_set):
        
#            rand_ind = rand.randint(len(subcorp_set))
            print(i, end="\t")
#            
#            subcorp = subcorp_set[rand_ind]
#        
            vocab_sizes = []
            for n in ntoks:
                subsample = Sentences(sent_subsample(subcorp, n))

                vcb_len = compute_vocab_size(subsample)
#                print(vcb_len, end=" ", flush=True)

                vocab_sizes.append(vcb_len)
            plt.plot(ntoks, vocab_sizes, '--', color=colour, 
                     label=(name if i == 1 else None))
        
    plt.legend()
    plt.show()