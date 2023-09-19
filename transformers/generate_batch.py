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
    parser.add_argument("--data", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--setting", default=None)
    parser.add_argument("--src_lang", default=None)
    parser.add_argument("--tgt_lang", default=None)
    parser.add_argument("--batch", type=int, default=None)

    args = parser.parse_args()

    model = MBartForConditionalGeneration.from_pretrained(args.model).to("cuda")
    tokenizer = MyMBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", src_lang="en_XX", tgt_lang="en_XX")

    datas = load_json(args.data)
    if len(datas) % args.batch == 0:
        batch_num = (len(datas) // args.batch)
    else:
        batch_num = (len(datas) // args.batch) + 1

    f = codecs.open(args.model + "-{}-{}-{}_summary.txt".format(args.setting, args.src_lang, args.tgt_lang), "w",
                    "utf-8")

    summaries = []

    dialogues = []
    for data in datas:
        dialogues.append(data["dialogue"])

    for batch_id in tqdm(range(batch_num)):
        tmp_input = dialogues[batch_id * args.batch:(batch_id + 1) * args.batch]
        tmp_transform_inputs = tokenizer(tmp_input, max_length=1024, padding=True, truncation=True,
                                         add_special_tokens=False,
                                         return_tensors='pt').to("cuda")

        tmp_summary_ids = model.generate(tmp_transform_inputs['input_ids'], no_repeat_ngram_size=3, max_length=150,
                                         min_length=15,
                                         length_penalty=1.0, num_beams=5, early_stopping=True, repetition_penalty=1.4,
                                         forced_bos_token_id=tokenizer.lang_code_to_id[args.tgt_lang])

        summaries.extend([each_res_seq for each_res_seq in
                          tokenizer.batch_decode(tmp_summary_ids, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=True)])
    for summary in summaries:
        f.write(summary + "\n")
