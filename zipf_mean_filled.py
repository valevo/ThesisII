from data.reader import wiki_from_pickles
from data.corpus import Words, Sentences, Articles

from stats.stat_functions import compute_ranks, compute_freqs,\
                                compute_normalised_freqs, merge_to_joint,\
                                reduce_pooled
                    
from jackknife.plotting import hexbin_plot, plot_preds

from stats.mle import Mandelbrot

from collections import Counter
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt

import argparse


def parse_args():
    p = argparse.ArgumentParser()
    
    p.add_argument("--lang", type=str)
    
    args = p.parse_args()
    return args.lang


def join_stats(stat_ls):
    stats_joined = {}
    
    for stat_d in stat_ls:
        for w, s in stat_d.items():
            if not w in stats_joined:
                stats_joined[w] = [s]
            else:
                stats_joined[w].append(s)
                
    return stats_joined

def pool_ranks(stat_ls, join_func=set.union):
    common_types = join_func(
            *[set(stat_d.keys()) for stat_d in stat_ls]
            )   
    
    stats_joined = {w: [] for w in common_types}
    for stat_d in stat_ls:
        final_r = len(stat_d)
        for w in common_types:
            if w in stat_d:
                stats_joined[w].append(stat_d[w])
            else:
                stats_joined[w].append(final_r)
                final_r += 1
    return stats_joined



def pool_freqs(stat_ls, join_func=set.union):
    common_types = join_func(
            *[set(stat_d.keys()) for stat_d in stat_ls]
            )
    
    return {w: [d[w] if w in d else rand.random() for d in stat_ls]
            for w in common_types}


if __name__ == "__main__":
    lang = parse_args()
    d = "results/" + lang + "/plots/"
    wiki = list(wiki_from_pickles("data/" + lang + "_pkl", n_tokens=int(10e6)))
    n = int(1e6)
    m = 10
    
    subsamples1 = (Sentences.subsample(wiki, n) for _ in range(m))
    
    ranks = [compute_ranks(sub) for sub in subsamples1]
    ranks_joined = pool_ranks(ranks)
    
    mean_ranks = reduce_pooled(ranks_joined)
        
    print("ranks done")
    
    subsamples2 = (Sentences.subsample(wiki, n) for _ in range(m))

    freqs = [compute_freqs(sub) for sub in subsamples2]
    freqs_joined = pool_freqs(freqs)
    
    mean_freqs = reduce_pooled(freqs_joined)
    
    mean_freqs_above_0 = {w: mf for w, mf in mean_freqs.items() if mf >= 1}
    
    print("freqs done")
    
    joints = merge_to_joint(mean_ranks, mean_freqs)
    xs, ys = list(zip(*joints.values()))
    
    hexbin_plot(xs, ys)
    
    plt.title("filled")
    plt.show()
