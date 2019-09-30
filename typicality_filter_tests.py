# -*- coding: utf-8 -*-

from data.reader import wiki_from_pickles
from data.corpus import Sentences, Articles

from stats.stat_functions import compute_ranks, compute_freqs,\
    merge_to_joint
    
from stats.mle import Mandelbrot

from stats.entropy import mandelbrot_entropy, typicality


import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt


def construct_typical_set(zipf_model, corpus, m, k):
    rs, fs = compute_ranks(corpus), compute_freqs(corpus)
    joints = merge_to_joint(rs, fs)
    auto_typicality = typicality(zipf_model, joints)
    
    typicalities = []
    
    for i in range(k):
        sub2 = Sentences.subsample(corpus, k)
        sub_freqs = compute_freqs(sub2)
        sub_joints = merge_to_joint(rs, sub_freqs)

        sub_typicality = typicality(zipf_model, sub_joints)
        corrected_typicality = sub_typicality - auto_typicality
        typicalities.append(corrected_typicality)
        
    mean_typ, std_typ = np.mean(typicalities), np.var(typicalities)**.5
    
#    if mean_typ - std_typ < 0:
#        raise ValueError("something's weird... epsilon less than 0")
    
    return mean_typ + std_typ if mean_typ >= 0 else mean_typ - std_typ


def filter_typicality(sentences, zipf_model, rank_dict, epsilon, n):
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
            
            
if __name__ == "__main__":      
    wiki = list(wiki_from_pickles("data/ALS_pkl"))
    
    n = 3e6
    k = 1e4
    
    m = 100
    j = 50
    
    big_ranks = compute_ranks(Sentences.subsample(wiki, n))
    freqs = compute_freqs(Sentences.subsample(wiki, n))
    
    joint = merge_to_joint(big_ranks, freqs)

    xs, ys = list(zip(*sorted(joint.values())))

    mandelbrot = Mandelbrot(ys, xs)
    mandelbrot_fit = mandelbrot.fit(start_params=np.asarray([1.0, 1.0]), 
                                    method="powell", full_output=True)    
    mandelbrot.register_fit(mandelbrot_fit)
    mandelbrot.print_result()
    
    auto_typicality = typicality(mandelbrot, joint)
    
    print("Auto-Typicality: ", auto_typicality, "\n")
    
#    plt.loglog(xs, ys, '.', color="blue")
#    
#    plt.loglog(xs, mandelbrot.predict(mandelbrot.optim_params, n_obs=3e6),
#               '--', color="red")
#    
#    plt.show()
    
    
    typicalities = []
    
    for i in range(m):
        sub2 = Sentences.subsample(wiki, k)
            
        sub_freqs = compute_freqs(sub2)
        sub_joints = merge_to_joint(big_ranks, sub_freqs)

        sub_typicality = typicality(mandelbrot, sub_joints)
        
        typicalities.append(sub_typicality)
        
        print("typicality: ", sub_typicality)
    
    mean_typ, std_typ = np.mean(typicalities), np.var(typicalities)**.5    
    
    corrected_typicalities = np.asarray(typicalities) - auto_typicality
    corrected_mean, corrected_std = np.mean(corrected_typicalities),\
                                    np.var(corrected_typicalities)**.5    


    epsilon = corrected_mean + corrected_std if corrected_mean >= 0 else corrected_mean - corrected_std
    print(corrected_mean)
    print(epsilon)                                    
                                    
#    plt.hist(corrected_typicalities, bins=m//10)
#    plt.axvline(x=corrected_mean, ymin=0, ymax=m//5, color="green")
#    plt.axvline(x=corrected_mean + corrected_std, ymin=0, ymax=m//5, color="green")
#    plt.axvline(x=corrected_mean - corrected_std, ymin=0, ymax=m//5, color="green")        
#    plt.title("r from big C")
#    plt.show()  
    


    
#    typical = Sentences.subsample(wiki, 5e3)
#                
#    typ_freqs = compute_freqs(typical)
#    typ_joints = merge_to_joint(big_ranks, typ_freqs)
#    
#    typ_xs, typ_ys = list(zip(*sorted(typ_joints.values())))
#    
#    plt.loglog(typ_xs, typ_ys, '.')
#    plt.show()





    Sents = Sentences([s for a in wiki for s in a])
    colors = ["purple", "blue", "red", "orange", "green", "yellow"]
    for c, i in zip(colors, np.arange(1, 6, 1.0)):

        atypical = list(filter_typicality(Sents, mandelbrot, 
                                          big_ranks, i*epsilon, n // 25))
        
        atypical = Sentences(atypical)
        
        atypical_freqs = compute_freqs(atypical)
        atypical_joints = merge_to_joint(big_ranks, atypical_freqs)

        atypical_typicality = typicality(mandelbrot, atypical_joints)
        
        print("\n\n")
        print(atypical_typicality)
        print(atypical_typicality - auto_typicality)
        
        
        a_xs, a_ys = list(zip(*sorted(atypical_joints.values())))
        plt.loglog(a_xs, a_ys, '.', color=c, label=str(i*epsilon))
        
        
    plt.legend()
    plt.show()
        
        
    
    
    
    