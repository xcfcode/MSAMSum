batch=8
model=ckpt/m2m/checkpoint-23883
setting=m2m


CUDA_VISIBLE_DEVICES=0 python generate_all2all.py \
    --model $model \
    --setting $setting \
    --batch $batch
