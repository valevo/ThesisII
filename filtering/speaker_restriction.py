# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as rand

from functools import reduce



def filter_speaker_restrict(sents, n, history_len):
    cur_sample = rand.randint(len(sents))
    sampled_s = sents[cur_sample]
    
    yield sampled_s
    
    cur_hist = [cur_sample]
    used = {cur_sample}
    sampled = len(sampled_s)
    
    while sampled < n:
        cur_sample = rand.randint(len(sents))
        sampled_s = sents[cur_sample]
        
        if not sampled_s:
            continue
        
        if cur_sample in used:
            continue
        
        cur_disallowed = reduce(np.union1d, cur_hist)
        
        if np.intersect1d(sampled_s, cur_disallowed).size > 0:
            continue
        
        cur_hist.append(sampled_s)
        if len(cur_hist) > history_len:
            cur_hist.pop(0)
        print("CUR_HIST", cur_hist, flush=True)
        
        used.add(cur_sample)
        sampled += len(sampled_s)
        
        yield sampled_s
    
    
    