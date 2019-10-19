# -*- coding: utf-8 -*-
from data.reader import wiki_from_pickles
from data.corpus import Words, Articles, Sentences

from stats.stat_functions import compute_ranks, compute_freqs,\
                compute_normalised_freqs, merge_to_joint, pool_zipf

from jackknife.plotting import hexbin_plot

import matplotlib.pyplot as plt
import seaborn as sns


def convergence_main(wiki, rng, save_dir="./"):
    col_pal = sns.color_palette("bright") # "dark", "deep", "colorblind"
    for i, n in enumerate(rng):
        sub_i1 = Sentences.subsample(wiki, n)
        sub_i2 = Sentences.subsample(wiki, n)
        joint_i = merge_to_joint(compute_ranks(sub_i1),
                                     compute_freqs(sub_i2))
           
        xs, ys = list(zip(*joint_i.values()))

        hexbin_plot(xs, ys, xlbl="$\log~r(w)$", ylbl="$\log~f(w)$",
                    edgecolors=col_pal[i], color=col_pal[i],
                    label=str(n), alpha=1/(i+1)**.3, linewidths=1.0,
                    cbar=(True if i==0 else False))
    
    plt.legend()
    plt.savefig(save_dir + "convergence_" + "_".join(map(str, rng)) + ".png",
                dpi=300)
    
    
    for i, n in enumerate(rng):
        sub_i1 = Sentences.subsample(wiki, n)
        sub_i2 = Sentences.subsample(wiki, n)
        joint_i = merge_to_joint(compute_ranks(sub_i1),
                                     compute_normalised_freqs(sub_i2))
           
        xs, ys = list(zip(*joint_i.values()))
                        
        hexbin_plot(xs, ys, xlbl="$\log~r(w)$", ylbl="$\log~P(w)$",
                    edgecolors=col_pal[i], color=col_pal[i],
                    label=str(n), alpha=1/(i+1)**.3, linewidths=1.0,
                    cbar=(True if i==0 else False))
    
    plt.legend()
    plt.savefig(save_dir + "convergence_probs_" + "_".join(map(str, rng)) + ".png",
                dpi=300)



if __name__ == "__main__":
    d = "results/ALS/"
    
    wiki = list(wiki_from_pickles("data/ALS_pkl"))    
    rng = list(range(int(5e5), int(2.5e6)+1, int(5e5)))
    
    convergence_main(wiki, rng)


    