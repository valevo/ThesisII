# -*- coding: utf-8 -*-

from data.reader import wiki_from_pickles
from data.corpus import Sentences

from stats.stat_functions import compute_ranks, compute_freqs,\
    merge_to_joint
    

from filtering.speaker_restriction import filter_speaker_restrict


import matplotlib.pyplot as plt


if __name__ == "__main__":
    wiki = list(wiki_from_pickles("data/ALS_pkl"))
    Sents = Sentences([s for a in wiki for s in a])
    big_ranks = compute_ranks(Sentences.subsample(wiki, 3e6))
    
    
    n = 1e5
    
    colors = ["purple", "blue", "red", "orange", "green"]
    for c, hist_len in zip(colors, range(2, 20, 4)):
        filtered = list(filter_speaker_restrict(Sents, n, hist_len))
        
        print("Sampling done", n, hist_len)
        
        filt_sents = Sentences(filtered)
        filt_fs = compute_freqs(filt_sents)
        filt_joints = merge_to_joint(big_ranks, filt_fs)
        
        xs, ys = list(zip(*sorted(filt_joints.values())))
        plt.loglog(xs, ys, '.', color=c, label=str(hist_len))
        
        
    
    rand_sample = Sentences.subsample(wiki, n)
    
    rs, fs = compute_ranks(rand_sample), compute_freqs(rand_sample)
    js = merge_to_joint(big_ranks, fs)
        
    xs, ys = list(zip(*sorted(js.values())))
        
    plt.loglog(xs, ys, '.', color="black", label=str(n))
    
    plt.legend()
    
    plt.show()
        
        
    
    
    