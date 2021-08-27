import os

os.system("python PaddleSeg/train.py \
       --config config.yml \
       --do_eval \
       --use_vdl \
       --save_interval 100 \
       --log_iters 1 \
       --save_dir output \
       --iters 3000 \
       --keep_checkpoint_max 10 \
       --batch_size 12")