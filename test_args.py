# -*- coding: utf-8 -*-

import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", type=str)
    p.add_argument("--factors", nargs="*", type=int, default=[])
    p.add_argument("--hist_lens", nargs="*", type=int, default=[])

    
    args = p.parse_args()
    return args.lang, args.factors, args.hist_lens


if __name__ == "__main__":
    my_str, ls1, ls2 = parse_args()

    print("FROM test_args.py:")    
    print(my_str)
    print(ls1)
    print(ls2)
    print("=========================================")
    