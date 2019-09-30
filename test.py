# -*- coding: utf-8 -*-

from data.reader import wiki_from_pickles
from data.corpus import Articles, Sentences, Words


from stats.subsampling import subsample_words, subsample_sentences,\
        subsample_articles
from stats.stat_functions import compute_ranks, compute_freqs,\
    compute_normalised_freqs, merge_to_joint

from stats.mle import Mandelbrot
from stats.entropy import mandelbrot_entropy, empirical_entropy, neg_log_likelihood, typicality
        
import matplotlib.pyplot as plt
import numpy as np


    
if __name__ == "__main__":
    wiki = list(wiki_from_pickles("data/ALS_pkl"))
    
    Wiki = Articles(wiki)
    
    ranks = compute_ranks(Wiki)
    freqs = compute_freqs(Wiki)
    
    joint = merge_to_joint(ranks, freqs)

    xs, ys = list(zip(*joint.values()))

    mandelbrot = Mandelbrot(ys, xs)
    mandelbrot_fit = mandelbrot.fit(start_params=np.asarray([1.0, 1.0]), 
                                    method="powell", full_output=True)
        
    mandelbrot.register_fit(mandelbrot_fit)
    
    
    mandelbrot.print_result()
    
    
    mle_params = mandelbrot.optim_params
    
    
    probs = mandelbrot.prob(mle_params, ranks=xs)
    
    print(len(xs))
    
    print(len(probs))
    
    print(sorted(probs, reverse=True)[:10])
    
    print("\n\n")
    
    
    c1 = subsample_articles(wiki, 1e6)
    c2 = subsample_articles(wiki, 1e6)
    
    ranks1 = compute_ranks(c1)
    freqs1 = compute_freqs(c2)
    
    joint1 = merge_to_joint(ranks1, freqs1)
    
    print(mandelbrot_entropy(*mle_params))
    print(empirical_entropy(mandelbrot, joint1))
    print(typicality(mandelbrot, joint1))
    

    
    
    
#    colors = ["blue", "green", "orange", "yellow"]
#    
#    for i in range(4):
#    
#        c1 = subsample_articles(w, 1e6)
#        c2 = subsample_articles(w, 1e6)
#        
#        
#        ranks = compute_ranks(c1, unwrapper=tokens_from_wiki)
#        freqs = compute_freqs(c2, unwrapper=tokens_from_wiki)
#        
#        intersected_types = ranks.keys() & freqs.keys()
#        
#        points = [(ranks[w], freqs[w]) for w in intersected_types]
#        
#        
#        xs, ys = list(zip(*points))
#        
#        
#        plt.loglog(xs, ys, '.', color=colors[i], label=str(i))
#    
    

#    colors = ["blue", "green", "orange", "yellow"]
#
#    for i, c in zip([1e5, 3e5, 6e5, 1e6], colors):
#        
#        c1 = subsample_articles(wiki, i)
#        c2 = subsample_articles(wiki, i)
#        
#        arts1 = Articles(list(c1))
#        arts2 = Articles(list(c2))
#        
#        
#        ranks = compute_ranks(arts1)
#        freqs = compute_freqs(arts2)
#        
#        
#        intersected_types = ranks.keys() & freqs.keys()
#        
#        points = [(ranks[w], freqs[w]) for w in intersected_types]
#    
#        xs, ys = list(zip(*points))
#        plt.loglog(xs, ys, '.', color=c, label=str(i))
#    
#    
#    plt.legend()
#    plt.savefig("./zipf.png", dpi=300)
#    plt.close()