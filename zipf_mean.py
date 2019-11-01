from data.reader import wiki_from_pickles
from data.corpus import Words, Sentences, Articles

from stats.stat_functions import compute_ranks, compute_freqs,\
                                compute_normalised_freqs
                    
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

def merge_to_joint(ranks, freqs):
    common_types = ranks.keys() & freqs.keys()
    return {w : (ranks[w], freqs[w]) for w in common_types}

def pool_zipf(stat_ls):
    common_types = set.intersection(
            *[set(stat_d.keys()) for stat_d in stat_ls]
            )
    return {w: [stat_d[w] for stat_d in stat_ls] for w in common_types}    


def join_stats(stat_ls):
    stats_joined = {}
    
    for stat_d in stat_ls:
        for w, s in stat_d.items():
            if not w in stats_joined:
                stats_joined[w] = [s]
            else:
                stats_joined[w].append(s)
                
    return stats_joined


def join_stats_default(stat_ls, default_func):
    all_types = set.union(
            *[set(stat_d.keys()) for stat_d in stat_ls]
            )
    
    stats_joined = {w: [] for w in all_types}
    
    for w in all_types:
        for stat_d in stat_ls:
            if w in stat_d:
                stats_joined[w].append(stat_d[w])
            else:
                stats_joined[w].append(default_func(stat_d))
    
    return stats_joined


if __name__ == "__main__":
    lang = parse_args()
    d = "results/" + lang + "/plots/"
    wiki = list(wiki_from_pickles("data/" + lang + "_pkl", n_tokens=int(10e6)))
    n = int(1e6)
    m = 10
    
    LEVEL = Sentences
    
    subsamples1 = (LEVEL.subsample(wiki, n) for _ in range(m))
    
    ranks = [compute_ranks(sub) for sub in subsamples1]
    ranks_joined = join_stats(ranks)
    mean_ranks = {w: np.mean(r_ls) for w, r_ls in ranks_joined.items()}
            
    print(len(mean_ranks))
        
    
    subsamples2 = (LEVEL.subsample(wiki, n) for _ in range(m))
    
    freqs = [compute_freqs(sub) for sub in subsamples2]
    freqs_joined = join_stats(freqs)
    mean_freqs = {w: np.mean(f_ls) for w, f_ls in freqs_joined.items()}
    
    print(len(mean_freqs))
    

    joints = merge_to_joint(mean_ranks, mean_freqs)
    xs, ys = list(zip(*sorted(joints.values())))
    
    hexbin_plot(xs, ys, xlbl="$\log$ $\overline{r}(w)$", 
                ylbl="$\log$ $\overline{f}(w)$", label=str(m))
    
    
#    mandelbrot = Mandelbrot(ys, xs)
#    mandelbrot_fit = mandelbrot.fit(start_params=np.asarray([1.0, 1.0]), 
#                                    method="powell", full_output=True)    
#    mandelbrot.register_fit(mandelbrot_fit)
#    mandelbrot.print_result()
#    plot_preds(mandelbrot, np.asarray(xs))

    
    
    subsamples1 = [LEVEL.subsample(wiki, n)]
    
    ranks = [compute_ranks(sub) for sub in subsamples1]
    mean_ranks = {w: np.mean(rs)  for w, rs in pool_zipf(ranks).items()}
    
    print(len(mean_ranks))
    
    subsamples2 = [LEVEL.subsample(wiki, n)]
    
    freqs = [compute_freqs(sub) for sub in subsamples2]
    mean_freqs = {w: np.mean(fs)  for w, fs in pool_zipf(freqs).items()}
    
    print(len(mean_freqs))

    joints = merge_to_joint(mean_ranks, mean_freqs)
    xs, ys = list(zip(*sorted(joints.values())))
    
    hexbin_plot(xs, ys, xlbl="$\log$ $\overline{r}(w)$", 
                ylbl="$\log$ $\overline{f}(w)$",
                color="red", edgecolors="red", cmap="Reds_r", cbar=False,
                alpha=0.5, label=str(1))
    
    
#    all_sents = Sentences([s for a in wiki for s in a])
#    
#    all_ranks = compute_ranks(all_sents)
#    all_freqs = compute_normalised_freqs(all_sents)
#    
#    joints = merge_to_joint(all_ranks, all_freqs)
#    xs, ys = list(zip(*sorted(joints.values())))
#    
#    plt.loglog(xs, ys, '.', color="green")
    
    
    plt.legend()
    plt.title("Sampling level: " + LEVEL.__name__)
    plt.show()
