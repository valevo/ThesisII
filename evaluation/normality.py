# -*- coding: utf-8 -*-

from data.reader import wiki_from_pickles
from data.corpus import Sentences


from lexical_diversity import lex_div



if __name__ == "__main__":
    wiki = list(wiki_from_pickles("data/ALS_pkl"))
    
    m = 10
    n = int(5e5)
    
    
    
    unifs = [Sentences.subsample(wiki, n) for _ in range(m)]
    mtld_dist_unif = [lex_div.mtld(list(u.tokens())) for u in unifs] 
    
    