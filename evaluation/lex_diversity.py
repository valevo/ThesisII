# -*- coding: utf-8 -*-

from data.reader import wiki_from_pickles, corpora_from_pickles
from data.corpus import Sentences


from lexical_diversity import lex_div

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
    
    
def lex_div_dist_plots(tfs, srfs, unis, div_f, save_dir):
    hist_args = dict(alpha=1.0)
    for param, div_vals in tfs.items():
        sns.distplot(div_vals, label="TF " + str(param), hist_kws=hist_args)
    
    for param, div_vals in srfs.items():
        sns.distplot(div_vals, label="SRF " + str(param), hist_kws=hist_args)
        
    sns.distplot(unis, label="UNIF", axlabel=div_f.__name__, hist_kws=hist_args)
    
    plt.savefig(save_dir + div_f.__name__ + "_dist_plot.png", dpi=300)
    plt.close()
    
def lex_div_means(tfs, srfs, unis, div_f, save_dir):
    with open(save_dir + div_f.__name__ + "_means.txt", "w") as handle:
        for param, lex_div_ls in tfs.items():
            handle.write("TF " + str(param) + "\t")
            handle.write(str(np.mean(lex_div_ls).round(3)))
            handle.write("\t" + str(np.sqrt(np.var(lex_div_ls)).round(3)))
        for param, lex_div_ls in srfs.items():
            handle.write("SRF " + str(param) + "\t")
            handle.write(str(np.mean(lex_div_ls).round(3)))
            handle.write("\t" + str(np.sqrt(np.var(lex_div_ls)).round(3)))
            
        handle.write("UNIF " + "\t")
        handle.write(str(np.mean(unis).round(3)))
        handle.write("\t" + str(np.sqrt(np.var(unis)).round(3))) 
            
    

    
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", type=str)
    p.add_argument("--factors", nargs="*", type=int, default=[])
    p.add_argument("--hist_lens", nargs="*", type=int, default=[])
    
    args = p.parse_args()
    return args.lang, args.factors, args.hist_lens


def get_filters(filter_dir, k, names, param_name, param_ls):
    filters_dict = {}
    
    for param in param_ls:
        all_samples = corpora_from_pickles(filter_dir, names=names)
        cur_param_filters = [Sentences(c) for name_d, c in all_samples if 
                             name_d["k"] == k and name_d[param_name] == param]
        filters_dict[param] = cur_param_filters
        
    return filters_dict

def lex_div_main(tfs, srfs, unis, results_d):
    factors = sorted(tfs.keys())
    hist_lens = sorted(srfs.keys())
    half_factors = factors[1::2]
    half_tfs = {k: tfs[k] for k in half_factors}
    half_hist_lens = hist_lens[1::2]
    half_srfs = {k: srfs[k] for k in half_hist_lens}
    
    
    cutoff = int(1e5)
    for div_f in [lex_div.mtld, lex_div.hdd]:
        print("\nlex div with " + div_f.__name__, flush=True)
    
        tf_mtlds = {param: [div_f(list(s.tokens())[:cutoff]) for s in samples]
                    for param, samples in half_tfs.items()}
        print("done with ", div_f.__name__, " for TF", flush=True)
        srf_mtlds = {param: [div_f(list(s.tokens())[:cutoff]) for s in samples]
                    for param, samples in half_srfs.items()}
        print("done with ", div_f.__name__, " for SRF", flush=True)
        uni_mtlds = [div_f(list(s.tokens())[:cutoff]) for s in unis]
        print("done with ", div_f.__name__, " for UNI", flush=True)
        
        lex_div_dist_plots(tf_mtlds, srf_mtlds, uni_mtlds, div_f, save_dir=results_d)
        lex_div_means(tf_mtlds, srf_mtlds, uni_mtlds, save_dir=results_d)
    
    
