root=/users7/xiachongfeng/acl/acl22/multilingualDS/final/dialogues

data=$root/fr_dialogues.json
batch=8
model=ckpt/o2o/fr-fr/checkpoint-5308
setting=o2o
src_lang=fr_XX
tgt_lang=fr_XX

# ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN

CUDA_VISIBLE_DEVICES=0 python generate_batch.py \
    --data $data \
    --model $model \
    --setting $setting \
    --src_lang $src_lang \
    --tgt_lang $tgt_lang \
    --batch $batch
