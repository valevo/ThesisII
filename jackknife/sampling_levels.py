# -*- coding: utf-8 -*-
from data.reader import wiki_from_pickles
from data.corpus import Words, Articles, Sentences

from stats.stat_functions import compute_ranks, compute_freqs,\
                            merge_to_joint

from jackknife.plotting import hexbin_plot

import matplotlib.pyplot as plt
import seaborn as sns


#def hexbin_plot_sns(xs, ys, grid=None, **params):
#    
#    def remove_zeros(x_vals, y_vals):
#        return list(zip(*[(x, y) for x, y in zip(x_vals, y_vals) 
#                          if x > 0 and y > 0]))
#    
#    xs, ys = remove_zeros(xs, ys)
#
#    if not grid:
#        grid = sns.JointGrid(x=xs, y=ys)
#    hb = grid.plot_joint(plt.hexbin, cmap="Blues_r",
#                    gridsize=75, mincnt=1, bins="log",
#                    xscale="log", yscale="log")
#    plt.colorbar(hb)
#    plt.xlabel("$\log~r(w)$")


def sampling_levels_main(wiki, n):    
    art_subsample1 = Articles.subsample(wiki, n)
    art_subsample2 = Articles.subsample(wiki, n)
    
#    sent_subsample1 = Sentences.subsample(wiki, n)
#    sent_subsample2 = Sentences.subsample(wiki, n)
    
    word_subsample1 =  Words.subsample(wiki, n)
    word_subsample2 =  Words.subsample(wiki, n)
    
    
    art_ranks = compute_ranks(art_subsample1)
    art_freqs = compute_freqs(art_subsample2)
    art_joint = merge_to_joint(art_ranks, art_freqs)
    xs, ys = list(zip(*sorted(art_joint.values())))
    
    hexbin_plot(xs, ys, xlbl="$\log~r(w)$", ylbl="$\log~f(w)$",
                label="articles")
    
#    sent_ranks = compute_ranks(sent_subsample1)
#    sent_freqs = compute_freqs(sent_subsample2)
#    sent_joint = merge_to_joint(sent_ranks, sent_freqs)
#    xs, ys = list(zip(*sorted(sent_joint.values())))
#    
#    hexbin_plot(xs, ys, xlbl="$\log~r(w)$", ylbl="$\log~f(w)$", 
#                color="green", edgecolors="green", cmap="Greens_r",
#                label="sentences", cbar=False)
    
    
    word_ranks = compute_ranks(word_subsample1)
    word_freqs = compute_freqs(word_subsample2)
    word_joint = merge_to_joint(word_ranks, word_freqs)
    xs, ys = list(zip(*sorted(word_joint.values())))
    
    hexbin_plot(xs, ys, xlbl="$\log~r(w)$", ylbl="$\log~f(w)$", 
                color="red", edgecolors="red", cmap="Reds_r",
                label="words", cbar=False)
    
    plt.legend()
    plt.show()


    freq_joint = merge_to_joint(art_freqs, word_freqs)
    xs, ys = list(zip(*sorted(freq_joint.values())))
    
    hexbin_plot(xs, ys, 
                xlbl="$\log~f(w)$ from articles", 
                ylbl="$\log~f(w)$ from words",
                edgecolors="blue", cmap="Blues_r", linewidths=0.2,
                label="articles")    
    plt.show()


if __name__ == "__main__":
    d = "results/ALS/"
    n = int(1e6)
    
    wiki = list(wiki_from_pickles("data/ALS_pkl"))
    
    art_subsample1 = Articles.subsample(wiki, n)
    art_subsample2 = Articles.subsample(wiki, n)
    
    word_subsample1 =  Words.subsample(wiki, n)
    word_subsample2 =  Words.subsample(wiki, n)
    
    
    art_ranks = compute_ranks(art_subsample1)
    art_freqs = compute_freqs(art_subsample2)
    art_joint = merge_to_joint(art_ranks, art_freqs)
    xs, ys = list(zip(*sorted(art_joint.values())))
    
    hexbin_plot(xs, ys, xlbl="$\log~r(w)$", ylbl="$\log~f(w)$",
                label="articles")
    
    
    word_ranks = compute_ranks(word_subsample1)
    word_freqs = compute_freqs(word_subsample2)
    word_joint = merge_to_joint(word_ranks, word_freqs)
    xs, ys = list(zip(*sorted(word_joint.values())))
    
    hexbin_plot(xs, ys, xlbl="$\log~r(w)$", ylbl="$\log~f(w)$", 
                color="red", edgecolors="red", cmap="Reds_r",
                label="words", cbar=False)
    
    
    plt.legend()
    plt.show()
    
    
    freq_joint = merge_to_joint(art_freqs, word_freqs)
    
    xs, ys = list(zip(*sorted(freq_joint.values())))
    
    hexbin_plot(xs, ys, 
                xlbl="$\log~f(w)$ from articles", 
                ylbl="$\log~f(w)$ from words",
                edgecolors="blue", cmap="Blues_r", linewidths=0.2,
                label="articles")
    
    # plt.plot(list(range(1, 50000)), list(range(1, 50000)), '--', color="grey")
    
    plt.show()
    
    
    
    