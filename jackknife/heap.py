# -*- coding: utf-8 -*-

from data.reader import wiki_from_pickles
from data.corpus import Words, Articles, Sentences

from stats.stat_functions import compute_vocab_size

from jackknife.plotting import hexbin_plot

import numpy as np
import numpy.random as rand

import matplotlib.pyplot as plt
import seaborn as sns

import pickle
        
def heap(corp, rng):
    vocab_sizes = []
    for i, ntoks in enumerate(rng):
        if i % 10 == 0:
            print(i, ntoks)
        subsample = Sentences.subsample(corp, ntoks)
        vocab_size = compute_vocab_size(subsample)
        vocab_sizes.append(vocab_size)
        
    return vocab_sizes
        
        
def heap_main(wiki, rng, save_dir="./"):
    m = 20
    long_rng = np.tile(rng, m)
    vocab_sizes = [heap(wiki, rng) for _ in range(m)]
    all_sizes = [v_n for size_ls in vocab_sizes for v_n in size_ls]
        
    hexbin_plot(long_rng, all_sizes, xlbl="$n$", ylbl="$V(n)$",
                log=False, ignore_zeros=False, label="pooled")
    
    hexbin_plot(rng, vocab_sizes[0], xlbl="$n$", ylbl="$V(n)$",
                log=False, ignore_zeros=False, label="single",
                color="red", edgecolors="red", cmap="Reds_r", cbar=False)
    
    plt.legend()
    plt.savefig(save_dir + "vocab_growth_" + 
                str(min(rng)) + "_" + str(max(rng)) + "_" + str(len(rng)) + ".png",
                dpi=300)
    plt.close()
    
    with open(save_dir + "vocab_growth_" + 
              str(min(rng)) + "_" + str(max(rng)) + "_" + str(len(rng)) + 
              ".pkl", "wb") as handle:
        pickle.dump(vocab_sizes, handle)
    
    

if __name__ == "__main__":
    wiki = list(wiki_from_pickles("data/ALS_pkl"))
    
    rng = list(range(int(0), int(2e6)+1, int(2e3)))
    rng = list(range(int(0), int(2e4)+1, int(2e2)))
    
    vocab_sizes = heap(wiki, rng)
    
    hexbin_plot(rng, vocab_sizes, xlbl="$n$", ylbl="$V(n)$",
                log=False, ignore_zeros=False)
    
    plt.show()
    
    
    
    m = 20
    long_rng = np.tile(rng, m)
    vocab_sizes = [heap(wiki, rng) for _ in range(m)]
    all_sizes = [v_n for size_ls in vocab_sizes for v_n in size_ls]
    
    print(len(long_rng), len(all_sizes))
    
    hexbin_plot(long_rng, all_sizes, xlbl="$n$", ylbl="$V(n)$",
                log=False, ignore_zeros=False, label="pooled")
    
    hexbin_plot(rng, vocab_sizes[0], xlbl="$n$", ylbl="$V(n)$",
                log=False, ignore_zeros=False, label="single",
                color="red", edgecolors="red", cmap="Reds_r", cbar=False)
    
    plt.legend()
    plt.show()
    
    
    
    
    