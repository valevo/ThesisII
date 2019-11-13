# -*- coding: utf-8 -*-

from data.reader import wiki_from_pickles, corpora_from_pickles
from data.corpus import Sentences

from evaluation.jaccard import number_words, number_sents, jaccard 

from itertools import combinations

import matplotlib.pyplot as plt
import seaborn as sns

import argparse

def subcorps_to_jaccard(subcorp_ls, label_f):
    labeled_subcorps = [[label_f(s) for s in subcorp.sentences() if "".join(s)] 
                            for subcorp in subcorp_ls]
    combs = list(combinations(range(len(subcorp_ls)), 2))
    return [jaccard(labeled_subcorps[i], labeled_subcorps[j]) for i, j in combs]

def get_within_jaccards(tfs, srfs, unis, label_f):
    tf_jaccards = {param: subcorps_to_jaccard(sample_ls, label_f)
                    for param, sample_ls in tfs.items()}
    srf_jaccards = {param: subcorps_to_jaccard(sample_ls, label_f)
                    for param, sample_ls in srfs.items()}
    uni_jaccards = subcorps_to_jaccard(unis, label_f)
    
    return tf_jaccards, srf_jaccards, uni_jaccards


def within_jaccard_plots(tfs, srfs, unis, save_dir):
    hist_args = dict(alpha=1.0)
    for param, jaccard_vals in tfs.items():
        sns.distplot(jaccard_vals, label="TF " + str(param), hist_kws=hist_args)
    for param, jaccard_vals in srfs.items():
        sns.distplot(jaccard_vals, label="SRF " + str(param), hist_kws=hist_args)    
        
    sns.distplot(unis, label="UNIF", hist_kws=hist_args) 
    
    plt.legend()
    plt.savefig(save_dir + "jaccard_within.png", dpi=300)
    plt.close()



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

    wiki = list(wiki_from_pickles("data/" + lang + "_pkl"))
    sent_d, label_f = number_sents((s for a in wiki for s in a))
    word_d, word_label_f = number_words((w for a in wiki for s in a for w in s))


    k = 1000000    
    srfs = get_filters(d + "SRF/", k, ["k", "h", "i"], "h", hist_lens)
    tfs = get_filters(d + "TF/", k, ["k", "f", "i"], "f", factors)
    unis = [Sentences(c) for _, c in corpora_from_pickles(d + "UNI", names=["k", "i"])]

    tf_jaccards, srf_jaccards, uni_jaccards = get_within_jaccards(tfs, srfs, unis,
                                                                  label_f)
    
    within_jaccard_plots(tf_jaccards, srf_jaccards, uni_jaccards,
                         save_dir=results_d)
    
    
