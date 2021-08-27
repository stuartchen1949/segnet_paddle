import os

os.system("python PaddleSeg/predict.py \
       --config config.yml \
       --model_path output_bs_8——pre/best_model/model.pdparams \
       --image_path data/PaddleSeg/camvid/test \
       --save_dir output/result")

