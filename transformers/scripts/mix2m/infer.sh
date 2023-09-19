root=/users7/xiachongfeng/acl/acl22/multilingualDS/final/mix2m

data=$root/test.json
batch=4
model=ckpt/mix2m/checkpoint-15922
setting=mix2m

# ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
CUDA_VISIBLE_DEVICES=0 python generate_batch.py \
    --data $data \
    --model $model \
    --setting $setting \
    --src_lang en_XX \
    --tgt_lang es_XX \
    --batch $batch

CUDA_VISIBLE_DEVICES=0 python generate_batch.py \
    --data $data \
    --model $model \
    --setting $setting \
    --src_lang en_XX \
    --tgt_lang en_XX \
    --batch $batch

CUDA_VISIBLE_DEVICES=0 python generate_batch.py \
    --data $data \
    --model $model \
    --setting $setting \
    --src_lang en_XX \
    --tgt_lang zh_CN \
    --batch $batch

CUDA_VISIBLE_DEVICES=0 python generate_batch.py \
    --data $data \
    --model $model \
    --setting $setting \
    --src_lang en_XX \
    --tgt_lang ru_RU \
    --batch $batch

CUDA_VISIBLE_DEVICES=0 python generate_batch.py \
    --data $data \
    --model $model \
    --setting $setting \
    --src_lang en_XX \
    --tgt_lang ar_AR \
    --batch $batch

CUDA_VISIBLE_DEVICES=0 python generate_batch.py \
    --data $data \
    --model $model \
    --setting $setting \
    --src_lang en_XX \
    --tgt_lang fr_XX \
    --batch $batch