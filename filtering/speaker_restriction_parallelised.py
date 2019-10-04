# -*- coding: utf-8 -*-

import numpy as np
import numpy.random as rand

from functools import reduce


sep = "■"


def filter_speaker_restrict(sents, n, history_len):
    cur_sample = rand.randint(len(sents))
    sampled_s = sents[cur_sample].split(sep)
    yield sampled_s
    
    cur_hist = [sampled_s]
    used = {cur_sample}
    sampled = len(sampled_s)
    
    while sampled < n:
        cur_sample = rand.randint(len(sents))
        sampled_s = sents[cur_sample].split(sep)
        
        if not sampled_s:
            continue
        
        if cur_sample in used:
            continue
        
        cur_disallowed = reduce(np.union1d, cur_hist)
        
        if np.intersect1d(sampled_s, cur_disallowed).size > 0:
            continue
        
        if len(cur_hist) >= history_len:
            cur_hist.pop(0)
        cur_hist.append(sampled_s)
        
        used.add(cur_sample)
        sampled += len(sampled_s)
        
        yield sampled_s
    
    
    