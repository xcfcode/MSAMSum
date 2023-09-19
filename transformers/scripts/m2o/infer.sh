batch=8
model=ckpt/m2o/checkpoint-15922
setting=m2o


CUDA_VISIBLE_DEVICES=0 python generate_all2all.py \
    --model $model \
    --setting $setting \
    --batch $batch
