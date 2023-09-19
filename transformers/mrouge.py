# -*- coding: utf-8 -*-
# @Time    : 2021/11/2 9:42 上午
# @Author  : Xiachong Feng
# @File    : mrouge.py
# @Software: PyCharm
import argparse
import codecs
import os
from rouge_score import rouge_scorer
from tqdm import tqdm

# bengali, hindi, turkish, arabic, danish, dutch, english, finnish, french, german, hungarian, italian, norwegian, portuguese, romanian, russian, spanish, swedish
# chinese, thai, japanese

# python mrouge.py --c ckpt/zh-zh/checkpoint-3573/summary.txt --r mdata/zh/ref.txt --l chinese
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", default=None, help="candidate file")
    parser.add_argument("--r", default=None, help="reference file")
    parser.add_argument("--l", default=None, help="language")
    args = parser.parse_args()

    f_c = codecs.open(os.path.join("./transformers-acl22/", args.c), "r", "utf-8")
    candidates = f_c.readlines()
    f_r = codecs.open(os.path.join("./multilingualDS/final/summaries/", args.r), "r", "utf-8")
    references = f_r.readlines()
    assert len(candidates) == len(references)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True, lang=args.l)

    r1s = []
    r2s = []
    rls = []

    for candidate, reference in tqdm(zip(candidates, references)):
        # score(self, target, prediction)
        # https://github.com/csebuetnlp/xl-sum/blob/593e09ab228ac3fd59ff9091369c81a8a28e7be2/multilingual_rouge_scoring/rouge_scorer.py?_pjax=%23js-repo-pjax-container%2C%20div%5Bitemtype%3D%22http%3A%2F%2Fschema.org%2FSoftwareSourceCode%22%5D%20main%2C%20%5Bdata-pjax-container%5D#L175
        scores = scorer.score(reference, candidate)
        r1s.append(scores["rouge1"].fmeasure)
        r2s.append(scores["rouge2"].fmeasure)
        rls.append(scores["rougeL"].fmeasure)

    r1s_avg = sum(r1s) / len(r1s)
    r2s_avg = sum(r2s) / len(r2s)
    rls_avg = sum(rls) / len(rls)
    print("{} ROUGE: {:5.4f} {:5.4f} {:5.4f}".format(args.l, r1s_avg, r2s_avg, rls_avg))
