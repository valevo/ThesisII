# -*- coding: utf-8 -*-
from data.reader import wiki_from_pickles
from data.corpus import Words, Articles, Sentences

from stats.stat_functions import compute_ranks, compute_freqs,\
                        compute_normalised_freqs, merge_to_joint, pool_zipf

from jackknife.plotting import hexbin_plot

import matplotlib.pyplot as plt
import seaborn as sns


def format_scientific(n):
    formatted_s = "%.1e" % n
    formatted_s = formatted_s.replace("+0", "")
    formatted_s = formatted_s.replace("+", "")
    return formatted_s

def variance_within_size(wiki, n, save_dir):
    subsamples = [(Sentences.subsample(wiki, n), 
                   Sentences.subsample(wiki, n)) for _ in range(20)]
    
    joints = [merge_to_joint(compute_ranks(sub1),
                             compute_freqs(sub2)) for sub1, sub2 in subsamples]
    
    common_types = set.intersection(*map(lambda d: set(d.keys()), joints))
    
    r_f_pairs = [j[w] for w in common_types for j in joints]
    
    
    xs, ys = list(zip(*r_f_pairs))
    
    hexbin_plot(xs, ys, xlbl=r"$\log$ $r(w)$", ylbl=r"$\log$ $f(w)$", 
                label="pooled")
    
    
    subsample1 = Sentences.subsample(wiki, n)
    subsample2 = Sentences.subsample(wiki, n)
    ranks_single = compute_ranks(subsample1)
    freqs_single = compute_freqs(subsample2)
    joint_single = merge_to_joint(ranks_single, freqs_single)
    
    xs, ys = list(zip(*joint_single.values()))
    
    hexbin_plot(xs, ys, xlbl=r"$\log$ $r(w)$", ylbl=r"$\log$ $f(w)$",
                color="red", edgecolors="red", cmap="Reds_r", cbar=False,
                label="single")
    plt.legend()
    plt.savefig(save_dir + "variance_within_size_" + str(n) + ".png",
                dpi=300)
    plt.close()

def variance_across_size(wiki, n1, n2, save_dir):
    subsamples_small = [(Sentences.subsample(wiki, n1), 
                   Sentences.subsample(wiki, n1)) for _ in range(20)]
    
    joints = [merge_to_joint(compute_ranks(sub1),
                    compute_normalised_freqs(sub2)) for sub1, sub2 in subsamples_small]
    
    common_types = set.intersection(*map(lambda d: set(d.keys()), joints))
    
    r_f_pairs = [j[w] for w in common_types for j in joints]
    
    
    xs, ys = list(zip(*r_f_pairs))
    
    hexbin_plot(xs, ys, xlbl=r"$\log$ $r(w)$", ylbl=r"$\log$ $P(w)$", 
                label="pooled " + format_scientific(n1))
    
    
    subsamples_big = [(Sentences.subsample(wiki, n2), 
                   Sentences.subsample(wiki, n2)) for _ in range(20)]
    
    joints = [merge_to_joint(compute_ranks(sub1),
                    compute_normalised_freqs(sub2)) for sub1, sub2 in subsamples_big]
    
    common_types = set.intersection(*map(lambda d: set(d.keys()), joints))
    
    r_f_pairs = [j[w] for w in common_types for j in joints]
    
    
    xs, ys = list(zip(*r_f_pairs))
    
    hexbin_plot(xs, ys, xlbl=r"$\log$ $r(w)$", ylbl=r"$\log$ $P(w)$",
                color="red", edgecolors="red", cmap="Reds_r",
                cbar=False, label="pooled " + format_scientific(n2))
    
    
    plt.legend()
    plt.savefig(save_dir + "variance_across_size_" + str(n1) + "_" + str(n2) + ".png",
                dpi=300)
    plt.close()
    
def variance_main(wiki, n, small_n, big_n, save_dir="./"):
    variance_within_size(wiki, n, save_dir)
    variance_across_size(wiki, small_n, big_n, save_dir)
    

if __name__ == "__main__":
    d = "results/ALS/"
    n = int(1e6)
    n1 = int(5e5)
    n2 = int(3e6)
    
    wiki = list(wiki_from_pickles("data/ALS_pkl"))
    
    variance_main(wiki, n)

    
