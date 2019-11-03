# -*- coding: utf-8 -*-
from data.reader import wiki_from_pickles
from data.corpus import Words, Articles, Sentences

from stats.stat_functions import compute_ranks, compute_freqs,\
                compute_normalised_freqs, merge_to_joint

from jackknife.plotting import hexbin_plot, colour_palette

import matplotlib.pyplot as plt
import seaborn as sns

def format_scientific(n):
    formatted_s = "%.1e" % n
    formatted_s = formatted_s.replace("+0", "")
    formatted_s = formatted_s.replace("+", "")
    return formatted_s


def convergence_main(wiki, rng, save_dir="./"):
    for i, n in enumerate(rng):
        sub_i1 = Sentences.subsample(wiki, n)
        sub_i2 = Sentences.subsample(wiki, n)
        joint_i = merge_to_joint(compute_ranks(sub_i1),
                                     compute_freqs(sub_i2))
           
        xs, ys = list(zip(*joint_i.values()))

        hexbin_plot(xs, ys, xlbl=r"$\log$ $r(w)$", ylbl=r"$\log$ $f(w)$",
                    edgecolors=colour_palette[i], color=colour_palette[i],
                    label=format_scientific(n), 
                    alpha=1/(i+1)**.3, linewidths=1.0,
                    cbar=(True if i==0 else False))
    
    plt.legend()
    plt.savefig(save_dir + "convergence_" + "_".join(map(str, rng)) + ".png",
                dpi=300)
    plt.close()
    
    
    for i, n in enumerate(rng):
        sub_i1 = Sentences.subsample(wiki, n)
        sub_i2 = Sentences.subsample(wiki, n)
        joint_i = merge_to_joint(compute_ranks(sub_i1),
                                     compute_normalised_freqs(sub_i2))
           
        xs, ys = list(zip(*joint_i.values()))
                        
        hexbin_plot(xs, ys, xlbl=r"$\log$ $r(w)$", ylbl=r"$\log$ $P(w)$",
                    edgecolors=colour_palette[i], color=colour_palette[i],
                    label=str(n), alpha=1/(i+1)**.3, linewidths=1.0,
                    cbar=(True if i==0 else False))
    
    plt.legend()
    plt.savefig(save_dir + "convergence_probs_" + "_".join(map(str, rng)) + ".png",
                dpi=300)
    plt.close()




if __name__ == "__main__":
    d = "results/ALS/"
    
    wiki = list(wiki_from_pickles("data/ALS_pkl"))    
    rng = list(range(int(5e5), int(2.5e6)+1, int(5e5)))
    
    convergence_main(wiki, rng)


    