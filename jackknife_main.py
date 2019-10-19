# -*- coding: utf-8 -*-

from data.reader import wiki_from_pickles

from jackknife.sampling_levels import sampling_levels_main
from jackknife.variance import variance_main
from jackknife.convergence import convergence_main
from jackknife.heap import heap_main

if __name__ == "__main__":
    wiki = list(wiki_from_pickles("data/ALS_pkl"))
    
    # max data size / 2
    big_n = int(3e6)
    small_n = int(5e5)
    n = int(1e6)
    
    rng1 = list(range(int(5e5), int(2.5e6)+1, int(5e5)))
    
    sampling_levels_main(wiki, big_n)
    
    variance_main(wiki, n, big_n, small_n)
    
    convergence_main(wiki, rng1)
    
    rng2 = list(range(int(0), int(1e5)+1, int(2e3)))
    
    heap_main(wiki, rng2)