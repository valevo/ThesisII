# -*- coding: utf-8 -*-

from data.reader import wiki_from_pickles
from data.corpus import Sentences

from stats.stat_functions import compute_ranks, compute_freqs,\
    merge_to_joint
    
from stats.mle import Mandelbrot

from stats.entropy import typicality

from filtering.typicality import filter_typicality_incremental


import numpy as np
import matplotlib.pyplot as plt

def establish_typical_set(wiki, rank_dict, zipf_model, k, m):
    typicalities = []
    
    for i in range(m):
        sub = Sentences.subsample(wiki, k)
            
        sub_freqs = compute_freqs(sub)
        sub_joints = merge_to_joint(rank_dict, sub_freqs)

        sub_typicality = typicality(zipf_model, sub_joints)        
        typicalities.append(sub_typicality)
            
    mean_typ, std_typ = np.mean(typicalities), np.var(typicalities)**.5    
    
    return mean_typ, std_typ


def corrected_epsilon(mean_typ, std_typ, auto_typ):
    corrected_mean = mean_typ - auto_typ
    return corrected_mean + std_typ if corrected_mean >= 0\
            else corrected_mean - std_typ


if __name__ == "__main__":
    wiki = list(wiki_from_pickles("data/ALS_pkl"))
    sents = [s for a in wiki for s in a]
    big_ranks = compute_ranks(Sentences.subsample(wiki, 3e6))
    big_freqs = compute_freqs(Sentences.subsample(wiki, 3e6))
    
    big_joint = merge_to_joint(big_ranks, big_freqs)

    xs, ys = list(zip(*sorted(big_joint.values())))

    mandelbrot = Mandelbrot(ys, xs)
    mandelbrot_fit = mandelbrot.fit(start_params=np.asarray([1.0, 1.0]), 
                                    method="powell", full_output=True)    
    mandelbrot.register_fit(mandelbrot_fit)
    mandelbrot.print_result()
        
    auto_typ = typicality(mandelbrot, big_joint)
    
    print("a*:", auto_typ)
    
    n = 1e5

    corrected_epsilon = corrected_epsilon(
            *establish_typical_set(wiki, big_ranks, mandelbrot, n, 50),
            auto_typ)
    
    print("epsilon':", corrected_epsilon)
    
    
    
    colors = ["purple", "blue", "red", "orange", "green"]
    for c, factor in zip(colors, np.linspace(1, 16, num=6).astype("int")):
        filtered = list(filter_typicality_incremental(sents, mandelbrot, big_ranks, 
                                  auto_typ, factor*corrected_epsilon, n))
        
        print("Sampling done", n, factor)
        
        filt_sents = Sentences(filtered)
        filt_fs = compute_freqs(filt_sents)
        filt_joints = merge_to_joint(big_ranks, filt_fs)
        
        xs, ys = list(zip(*sorted(filt_joints.values())))
        plt.loglog(xs, ys, '.', color=c, 
                   label=str(round(factor*corrected_epsilon, 4)))
        
        
    
    rand_sample = Sentences.subsample(wiki, n)
    
    rs, fs = compute_ranks(rand_sample), compute_freqs(rand_sample)
    js = merge_to_joint(big_ranks, fs)
        
    xs, ys = list(zip(*sorted(js.values())))
        
    plt.loglog(xs, ys, '.', color="black", label=str(n))
    
    plt.legend()
    
    plt.show()
        
        
    
    
    