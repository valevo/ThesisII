# -*- coding: utf-8 -*-

from data.reader import wiki_from_pickles, corpora_from_pickles
from data.corpus import Sentences


from lexical_diversity import lex_div

import matplotlib.pyplot as plt
import seaborn as sns
    
    
def lex_div_dist_plots(tfs, srfs, unis, div_f, save_dir):
    hist_args = dict(alpha=1.0)
    for param, samples in tfs.items():
        div_vals = [div_f(list(s.tokens())) for s in samples]
        sns.distplot(div_vals, label="TF " + str(param), hist_kws=hist_args)
    
    for param, samples in srfs.items():
        div_vals = [div_f(list(s.tokens())) for s in samples]
        sns.distplot(div_vals, label="SRF " + str(param), hist_kws=hist_args)
        
    sns.distplot([div_f(list(s.tokens())) for s in unis], label="UNIF",
                  axlabel=div_f.__name__, hist_kws=hist_args)
    
    plt.savefig(save_dir + div_f.__name__ + "dist_plot.png", dpi=300)
    plt.close()

    
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

if __name__ == "__main__":    
    lang, factors, hist_lens = parse_args()
    print("ARGS: ", lang, factors, hist_lens, "\n")
    d =  "results/" + lang + "/"
    results_d = d + "evaluation/"

    k = 1000000
    
    srfs = get_filters(d + "SRF/", k, ["k", "h", "i"], "h", hist_lens)
    tfs = get_filters(d + "TF/", k, ["k", "f", "i"], "f", factors)
    unis = [Sentences(c) for _, c in corpora_from_pickles(d + "UNI", names=["k", "i"])]
    
    
    half_factors = factors[::2]
    half_tfs = {k: tfs[k] for k in half_factors}
    half_hist_lens = hist_lens[-2:]
    half_srfs = {k: srfs[k] for k in half_hist_lens}
    
    lex_div_dist_plots(tfs, srfs, unis, lex_div.mtld, save_dir=results_d)
    
    
    

    