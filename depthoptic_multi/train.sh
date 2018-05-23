#! /bin/bash
python depthoptic_main.py\
    --mode train\
    --model_name depthoptic_0.2\
    --datapath_prefix /home/derlee/boris/movie/\
    --datapath_file /home/derlee/depth_estimation/data/train_list.txt\
    --log_directory record/\
    --init_lr 1e-4\
    --batch_size 16\
    --gpus '1 2'\
    --input_height 128\
    --input_width 256\
    --num_epochs 200\
    --optic_reg_w 0.2\
    --train_network disnet\
    --build_network disnet\
