import argparse
import codecs
import os
from rouge_score import rouge_scorer
from tqdm import tqdm
import glob


def auto_infer(txt_file):
    segs = txt_file.split("-")
    setting = segs[2].strip()
    src_lang = segs[3].strip()

    if len(src_lang.split("_")) == 2:
        src_lang = src_lang.split("_")[0]

    tgt_lang = segs[4].split("_")[0].strip()

    return setting, src_lang, tgt_lang


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path2txt", default=None, help="path to txt files")
    args = parser.parse_args()

    path2txt = args.path2txt

    txt_files = glob.glob("{}/*.txt".format(path2txt))

    mrouge_lang_map = {"en": "english", "zh": "chinese", "es": "spanish", "ru": "russian", "ar": "arabic",
                       "fr": "french"}

    for txt_file in txt_files:

        setting, src_lang, tgt_lang = auto_infer(txt_file)

        src_file_path = txt_file  # candidate file
        print("src", src_file_path)

        ref_file_path = os.path.join("./multilingualDS/final/summaries/",
                                     "{}_ref.txt".format(tgt_lang))
        print("ref", ref_file_path)

        f_c = codecs.open(src_file_path, "r", "utf-8")
        candidates = f_c.readlines()

        f_r = codecs.open(ref_file_path, "r", "utf-8")
        references = f_r.readlines()

        assert len(candidates) == len(references)

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True,
                                          lang=mrouge_lang_map[tgt_lang])

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
        print("{}-{}-{} ROUGE: {:5.4f} {:5.4f} {:5.4f}".format(setting, src_lang, tgt_lang, r1s_avg, r2s_avg, rls_avg))
