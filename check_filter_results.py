# -*- coding: utf-8 -*-

from data.reader import wiki_from_pickles
from data.corpus import Sentences

from stats.stat_functions import compute_ranks, compute_freqs, merge_to_joint

import os
import pickle
import matplotlib.pyplot as plt


if __name__ == "__main__":
    n = int(2.5e6)
    lang_dir = "results/FI/"
    
#    # uniform subsamples
#    wiki = list(wiki_from_pickles("data/ALS_pkl"))
#    big_ranks = compute_ranks(Sentences.subsample(wiki, 3e6))
#    
#    sub1 = Sentences.subsample(wiki, n)
#    sub2 = Sentences.subsample(wiki, n)
#    
#    uni_rs, uni_fs = compute_ranks(sub1), compute_freqs(sub2)
#    uni_joints = merge_to_joint(uni_rs, uni_fs)
    
    
    
    # SRF
#    hist_lens = list(map(str, [2, 4, 6, 10, 14]))
#    srfs0 = dict()
#    for h in hist_lens:
#        with open(lang_dir + "SRF/" + str(n) + "_" + h + "_0.pkl", "rb") as handle:
#            srfs0[h] = pickle.load(handle)
#    srfs1 = dict()
#    for h in hist_lens:
#        with open(lang_dir + "SRF/" + str(n) + "_" + h + "_1.pkl", "rb") as handle:
#            srfs1[h] = pickle.load(handle)
#        
#    colors = ["purple", "blue", "red", "orange", "green", "yellow"]
#    for c, (k, v) in zip(colors, srfs0.items()):
#        srf_ranks = compute_ranks(Sentences(srfs1[k]))
#        srf_freqs = compute_freqs(Sentences(v))
#        srf_joints = merge_to_joint(srf_ranks, srf_freqs)
#        
#        xs, ys = list(zip(*sorted(srf_joints.values())))
#        
#        plt.loglog(xs, ys, '.', color=c, label=k)
#        
#    
#    xs, ys = list(zip(*sorted(uni_joints.values())))
#    
#    plt.loglog(xs, ys, '.', color="black")
#    
#    plt.legend()
#    plt.title("SRF Results")
#    plt.show()
    
    
    # TF
    factors = list(map(str, [500.0]))
    tfs0 = dict()
    for f in factors:
        with open(lang_dir + "TF/" + str(n) + "_" + f + "_0.pkl", "rb") as handle:
            tfs0[f] = pickle.load(handle)
    tfs1 = dict()
    for f in factors:
        with open(lang_dir + "TF/" + str(n) + "_" + f + "_1.pkl", "rb") as handle:
            tfs1[f] = pickle.load(handle)
        
    colors = ["purple", "blue", "red", "orange", "green", "yellow"]
    for c, (k, v) in zip(colors, tfs0.items()):
        tf_ranks = compute_ranks(Sentences(tfs1[k]))
        tf_freqs = compute_freqs(Sentences(v))
        tf_joints = merge_to_joint(tf_ranks, tf_freqs)
        
        xs, ys = list(zip(*sorted(tf_joints.values())))
        
        plt.loglog(xs, ys, '.', color=c, label=k)
        
    
#    xs, ys = list(zip(*sorted(uni_joints.values())))
    
#    plt.loglog(xs, ys, '.', color="black")
    
    plt.legend()
    plt.title("TF Results")
    plt.show()
    
    
    
    