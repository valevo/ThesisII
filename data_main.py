# -*- coding: utf-8 -*-

from data.reader import wiki_from_pickles
from data.corpus import Sentences

from stats.stat_functions import compute_ranks, compute_freqs,\
                    merge_to_joint, compute_vocab_size
                    
from stats.mle import Mandelbrot, Heap

from jackknife.plotting import hexbin_plot, plot_preds

from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

import argparse
import pickle
import os

def heap(corp, rng):
    vocab_sizes = []
    for i, ntoks in enumerate(rng):
        if i % 100 == 0:
            print(i, ntoks)
        subsample = Sentences.subsample(corp, ntoks)
        vocab_size = compute_vocab_size(subsample)
        vocab_sizes.append(vocab_size)
        
    return vocab_sizes

def heap_from_file(save_dir, rng_params):
    rng_params = map(str, rng_params)
    required_file_name = "vocab_growth_" + "_".join(rng_params) + ".pkl"
    print(required_file_name)
    if required_file_name in os.listdir(save_dir):
        with open(save_dir + required_file_name, "rb") as handle:
            return pickle.load(handle)
    else:
        raise FileNotFoundError


def parse_args():
    p = argparse.ArgumentParser()
    
    p.add_argument("--lang", type=str)
    
    args = p.parse_args()
    return args.lang

if __name__ == "__main__":
    lang = parse_args()
    d = "results/" + lang + "/plots/"
    wiki = list(wiki_from_pickles("data/" + lang + "_pkl"))
    n = int(25e6)
    
    subsample1 = Sentences.subsample(wiki, n)
    subsample2 = Sentences.subsample(wiki, n)
    
    print("samples done")
    ranks, freqs = compute_ranks(subsample1), compute_freqs(subsample2)
    joints = merge_to_joint(ranks, freqs)
    xs, ys = list(zip(*sorted(joints.values())))
    
    hexbin_plot(xs, ys, xlbl="$\log$ $r(w)$", ylbl="$\log$ $f(w)$")
    
    mandelbrot = Mandelbrot(ys, xs)
    mandelbrot_fit = mandelbrot.fit(start_params=np.asarray([1.0, 1.0]), 
                                    method="powell", full_output=True)    
    mandelbrot.register_fit(mandelbrot_fit)
    mandelbrot.print_result()
    with open(d + "mle_mandelbrot_" + str(n) + ".txt", "w") as handle:
        handle.write(mandelbrot.print_result(string=True))
    plot_preds(mandelbrot, np.asarray(xs))
    plt.savefig(d + "rank_freq_" + str(n) + ".png",
            dpi=300)
    plt.close()
    
    print("rank-freq relationship done")
    
    
    freq_of_freqs = Counter(freqs.values())
    xs, ys = list(zip(*freq_of_freqs.items()))
    
    hexbin_plot(xs, ys, xlbl="$\log$ $f$", ylbl="$\log$ $f(f)$")
    plt.savefig(d + "freq_freqs_" + str(n) + ".png",
            dpi=300)
    plt.close()
    
    print("freq-freq relationship done")
    
    
    
    start, end, step = 0, n+1, n//2000
    rng = list(range(start, end, step))
    try:
        v_ns = heap_from_file(d, (start, end, len(rng)))
    except FileNotFoundError:
        v_ns = heap(wiki, rng)
    
    hexbin_plot(rng, v_ns, xlbl="$n$", ylbl="$V(n)$", log=False, gridsize=100)
    
    heap = Heap(v_ns, rng)
    heap_fit = heap.fit(start_params=np.asarray([1.0, 1.0]), 
                                    method="powell", full_output=True)    
    heap.register_fit(heap_fit)
    heap.print_result()
    
    plot_preds(heap, np.asarray(rng))
    
    with open(d + "mle_heap_" + str(n) + ".txt", "w") as handle:
        handle.write(heap.print_result(string=True))
    
    with open(d + "vocab_growth_" + str(start) + "_" + 
                str(end) + "_" + str(len(rng)) + ".pkl", "wb") as handle:
        pickle.dump(v_ns, handle)
    
    plt.savefig(d + "vocab_growth_" + str(start) + "_" + 
                str(end) + "_" + str(len(rng)) + ".png",
                dpi=300)
    plt.close()
    
    print("heap done")
    
    
    
    