import argparse
import codecs
import json
import os
from tqdm import tqdm

from transformers import MBartTokenizer, MBartForConditionalGeneration, MBart50Tokenizer


def load_json(path):
    with codecs.open(path, "r", "utf-8") as f:
        datas = []
        for line in f:
            datas.append(json.loads(line))
    return datas


class MyMBart50Tokenizer(MBart50Tokenizer):

    def set_src_lang_special_tokens(self, src_lang) -> None:
        ...

    def set_tgt_lang_special_tokens(self, lang: str) -> None:
        ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--setting", default=None)
    parser.add_argument("--batch", type=int, default=None)

    args = parser.parse_args()

    """
    load model and tokenizer
    """
    model = MBartForConditionalGeneration.from_pretrained(args.model).to("cuda")
    tokenizer = MyMBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX",
                                                 tgt_lang="en_XX")

    """
    load dialogues
    """
    en_datas = load_json("./multilingualDS/final/dialogues/en_dialogues.json")
    zh_datas = load_json("./multilingualDS/final/dialogues/zh_dialogues.json")
    es_datas = load_json("./multilingualDS/final/dialogues/es_dialogues.json")
    ru_datas = load_json("./multilingualDS/final/dialogues/ru_dialogues.json")
    ar_datas = load_json("./multilingualDS/final/dialogues/ar_dialogues.json")
    fr_datas = load_json("./multilingualDS/final/dialogues/fr_dialogues.json")

    """
    pre-process for dialogues
    """
    en_dialogues = [en_data["dialogue"] for en_data in en_datas]
    zh_dialogues = [zh_data["dialogue"] for zh_data in zh_datas]
    es_dialogues = [es_data["dialogue"] for es_data in es_datas]
    ru_dialogues = [ru_data["dialogue"] for ru_data in ru_datas]
    ar_dialogues = [ar_data["dialogue"] for ar_data in ar_datas]
    fr_dialogues = [fr_data["dialogue"] for fr_data in fr_datas]
    dialogues = {"en": en_dialogues, "zh": zh_dialogues, "es": es_dialogues, "ru": ru_dialogues, "ar": ar_dialogues,
                 "fr": fr_dialogues}

    """
    Get Batch Num
    """
    if len(en_datas) % args.batch == 0:
        batch_num = (len(en_datas) // args.batch)
    else:
        batch_num = (len(en_datas) // args.batch) + 1

    """
    All settings
    """
    m2m_pairs = [{"src": "en", "to": "en"}, {"src": "en", "to": "es"}, {"src": "en", "to": "ru"},
                 {"src": "en", "to": "fr"}, {"src": "en", "to": "ar"}, {"src": "en", "to": "zh"},
                 {"src": "es", "to": "en"}, {"src": "es", "to": "es"}, {"src": "es", "to": "ru"},
                 {"src": "es", "to": "fr"}, {"src": "es", "to": "ar"}, {"src": "es", "to": "zh"},
                 {"src": "ru", "to": "en"}, {"src": "ru", "to": "es"}, {"src": "ru", "to": "ru"},
                 {"src": "ru", "to": "fr"}, {"src": "ru", "to": "ar"}, {"src": "ru", "to": "zh"},
                 {"src": "fr", "to": "en"}, {"src": "fr", "to": "es"}, {"src": "fr", "to": "ru"},
                 {"src": "fr", "to": "fr"}, {"src": "fr", "to": "ar"}, {"src": "fr", "to": "zh"},
                 {"src": "ar", "to": "en"}, {"src": "ar", "to": "es"}, {"src": "ar", "to": "ru"},
                 {"src": "ar", "to": "fr"}, {"src": "ar", "to": "ar"}, {"src": "ar", "to": "zh"},
                 {"src": "zh", "to": "en"}, {"src": "zh", "to": "es"}, {"src": "zh", "to": "ru"},
                 {"src": "zh", "to": "fr"}, {"src": "zh", "to": "ar"}, {"src": "zh", "to": "zh"}]

    o2m_pairs = [{"src": "en", "to": "en"}, {"src": "en", "to": "es"}, {"src": "en", "to": "ru"},
                 {"src": "en", "to": "fr"}, {"src": "en", "to": "ar"}, {"src": "en", "to": "zh"}]

    m2o_pairs = [{"src": "en", "to": "en"}, {"src": "es", "to": "en"}, {"src": "ru", "to": "en"},
                 {"src": "fr", "to": "en"}, {"src": "ar", "to": "en"}, {"src": "zh", "to": "en"}]

    all_pairs = None
    if args.setting == "m2o":
        all_pairs = m2o_pairs
    elif args.setting == "o2m":
        all_pairs = o2m_pairs
    elif args.setting == "m2m":
        all_pairs = m2m_pairs

    """
    for each setting, test
    """
    mbart_map = {"zh": "zh_CN", "ar": "ar_AR", "ru": "ru_RU", "en": "en_XX", "es": "es_XX", "fr": "fr_XX"}

    for pair in all_pairs:
        print("{}-{}".format(pair["src"], pair["to"]))

        f = codecs.open(args.model + "-{}-{}-{}_summary.txt".format(args.setting, pair["src"], pair["to"]), "w",
                        "utf-8")

        summaries = []
        for batch_id in tqdm(range(batch_num)):
            tmp_input = dialogues[pair["src"]][batch_id * args.batch:(batch_id + 1) * args.batch]
            tmp_transform_inputs = tokenizer(tmp_input, max_length=1024, padding=True, truncation=True,
                                             add_special_tokens=False,
                                             return_tensors='pt').to("cuda")

            tmp_summary_ids = model.generate(tmp_transform_inputs['input_ids'], no_repeat_ngram_size=3, max_length=150,
                                             min_length=15,
                                             length_penalty=1.0, num_beams=5, early_stopping=True,
                                             repetition_penalty=1.4,
                                             forced_bos_token_id=tokenizer.lang_code_to_id[mbart_map[pair["to"]]])

            summaries.extend([each_res_seq for each_res_seq in
                              tokenizer.batch_decode(tmp_summary_ids, skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=True)])
        for summary in summaries:
            f.write(summary + "\n")

        f.close()
