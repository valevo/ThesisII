# -*- coding: utf-8 -*-

from data.reader import wiki_from_pickles
from data.corpus import Articles, Sentences, Words

from stats.stat_functions import compute_ranks, compute_freqs, merge_to_joint
from stats.mle import Mandelbrot
from stats.entropy import mandelbrot_entropy, neg_log_likelihood, empirical_entropy,\
            typicality

import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    wiki = list(wiki_from_pickles("data/ALS_pkl"))
    
    Wiki = Articles(wiki)
    
    print(len(list(Wiki.tokens())))
    
    ranks = compute_ranks(Sentences.subsample(wiki, 3e6))
    freqs = compute_freqs(Sentences.subsample(wiki, 3e6))
    
    joint = merge_to_joint(ranks, freqs)

    xs, ys = list(zip(*joint.values()))

    mandelbrot = Mandelbrot(ys, xs)
    mandelbrot_fit = mandelbrot.fit(start_params=np.asarray([1.0, 1.0]), 
                                    method="powell", full_output=True)    
    mandelbrot.register_fit(mandelbrot_fit)
    mandelbrot.print_result()
    
    print(mandelbrot_entropy(*mandelbrot.optim_params))

    auto_typ = typicality(mandelbrot, joint)

    print("auto-typicality:", auto_typ)
    
    print("\n\n")
    
    ns = [1e6]
    colors = ["blue", "green", "yellow", "red"]
    
    for n, c in zip(ns, colors):
        typicalities = []
        for i in range(20):   
    #        print("\n", i)
            sub1 = Sentences.subsample(wiki, n)
            sub2 = Sentences.subsample(wiki, n)
            
            sub_ranks = compute_ranks(sub1)
            sub_freqs = compute_freqs(sub2)
        
            sub_joint = merge_to_joint(sub_ranks, sub_freqs)
            
    #        print(empirical_entropy(mandelbrot, sub_joint))
            typ = typicality(mandelbrot, sub_joint)
            
            corrected_typ = typ - auto_typ
            
            print(round(typ, 5), round(corrected_typ, 5))
            
            
            typicalities.append(corrected_typ)
        
        
        min_typ, max_typ = min(typicalities), max(typicalities)
        mean, std = np.mean(typicalities), np.var(typicalities)**.5
        print("\t min, max", min_typ, max_typ)
        print("\t mean, std", mean, std)
        print(c)
        plt.hist(typicalities, bins=5, color=c, label=str(n))
    
    plt.axvline(mean+std, ymin=0, ymax=20, color="red")
    plt.axvline(mean-std, ymin=0, ymax=20, color="red")
    
    plt.axvline(max_typ, ymin=0, ymax=20, color="green")
    plt.axvline(min_typ, ymin=0, ymax=20, color="green")

    plt.legend()
    plt.show()