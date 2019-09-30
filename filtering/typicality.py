# -*- coding: utf-8 -*-

import numpy.random as rand

def filter_typicality(sentences, zipf_model, rank_dict, epsilon, n):
    sampled = 0
    
    sampled_sents = []
    
    while sampled < n:
        cur_sample = rand.randint(len(sentences))
        cur_sent = sentences.elements[cur_sample]
        
        
        tentative_sampled = Sentences(sampled_sents + [cur_sent])
        
        fs = compute_freqs(tentative_sampled)
        
        cur_joint = merge_to_joint(rank_dict, fs)
        
        
        #correct with auto-typicality and maybe abs
        if typicality(zipf_model, cur_joint) > epsilon:
            
            sampled_sents.append(cur_sent)
            yield cur_sent
    
            