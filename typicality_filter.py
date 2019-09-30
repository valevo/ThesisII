# -*- coding: utf-8 -*-

from data.corpus import Words, Sentences, Articles

from stats.stat_functions import compute_ranks, compute_freqs, merge_to_joint

from stats.mle import Mandelbrot

from stats.entropy import neg_log_likelihood, empirical_entropy, typicality

import numpy as np
import numpy.random as rand


def set_up_filtering(corpus, n):
    big_ranks = compute_ranks(Sentences.subsample(corpus, n))
    freqs = compute_freqs(Sentences.subsample(corpus, n))
    
    joint = merge_to_joint(big_ranks, freqs)

    xs, ys = list(zip(*sorted(joint.values())))

    mandelbrot = Mandelbrot(ys, xs)
    mandelbrot_fit = mandelbrot.fit(start_params=np.asarray([1.0, 1.0]), 
                                    method="powell", full_output=True)    
    mandelbrot.register_fit(mandelbrot_fit)
    mandelbrot.print_result()
    
    auto_typicality = typicality(mandelbrot, joint)    


    return big_ranks, mandelbrot, auto_typicality


    
    
def filter_typicality(sentences, zipf_model, rank_dict, auto_typicality, epsilon, n):
    sampled = 0
    sampled_sents = []       
    
    while sampled < n:
        cur_sample = rand.randint(len(sentences))
        cur_sent = sentences.elements[cur_sample]
        
        
        tentative_sampled = Sentences(sampled_sents + [cur_sent])
        
        fs = compute_freqs(tentative_sampled)
        cur_joint = merge_to_joint(rank_dict, fs)
        
        print(".", end="", flush=True)

        if typicality(zipf_model, cur_joint) - auto_typicality < epsilon:
            sampled += len(cur_sent)
            sampled_sents.append(cur_sent)
            yield cur_sent
            print(" ", 
                  round(typicality(zipf_model, cur_joint) - auto_typicality, 3), 
                  end=" ")  