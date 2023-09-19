batch=8
model=ckpt/o2m/checkpoint-15922
setting=o2m


CUDA_VISIBLE_DEVICES=0 python generate_all2all.py \
    --model $model \
    --setting $setting \
    --batch $batch
