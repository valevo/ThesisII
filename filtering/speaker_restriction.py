# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as rand

from functools import reduce



def filter_speaker_restrict(sentences, n, history_len):
    cur_sample = rand.randint(len(sentences))
    
    yield sentences.elements[cur_sample]
    
    cur_hist = [cur_sample]
    used = {cur_sample}
    sampled = len(sentences.elements[cur_sample])
    
    while sampled < n:
        cur_sample = rand.randint(len(sentences))
        sampled_s = sentences.elements[cur_sample]
        
        if not sampled_s:
            continue
        
        if cur_sample in used:
            continue
        
        cur_disallowed = reduce(np.union1d, cur_hist)
        
        if np.intersect1d(sampled_s, cur_disallowed).size > 0:
#            print(np.intersect1d(sampled_s, cur_disallowed))
            continue
        
        if len(cur_hist) >= history_len:
            cur_hist.pop(0)
        cur_hist.append(sampled_s)
        
        used.add(cur_sample)
        sampled += len(sampled_s)
        
        yield sampled_s
        
        print(".", end="")
    
    
    
    