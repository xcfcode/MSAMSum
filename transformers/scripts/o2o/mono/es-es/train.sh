root=./multilingualDS/final
train_file=$root/o2o/es-es/train.json
validation_file=$root/o2o/es-es/valid.json
output_dir=ckpt/o2o/es-es/

model=facebook/mbart-large-50-many-to-many-mmt

batch_size=4
lr=5e-06
warm_up=2000
train_epochs=20

CUDA_VISIBLE_DEVICES=0 python run_summarization.py \
    --model_name_or_path $model \
    --do_train \
    --do_eval \
    --num_train_epochs $train_epochs \
    --output_dir $output_dir \
    --train_file $train_file \
    --validation_file $validation_file \
    --text_column dialogue --summary_column summary \
    --per_device_train_batch_size=$batch_size \
    --per_device_eval_batch_size=$batch_size \
    --overwrite_output_dir \
    --learning_rate $lr --label_smoothing_factor 0.1 --weight_decay 0.01 --lr_scheduler_type polynomial --warmup_steps $warm_up \
    --save_strategy epoch --evaluation_strategy epoch \
    --seed 8888 \
    --fp16
