#! /bin/bash
python depthoptic_multi_main.py\
    --mode train\
    --model_name depthoptic_optic\
    --datapath_prefix /home/derlee/boris/movie/\
    --datapath_file /home/derlee/depth_estimation/data/train_list_tmp.txt\
    --log_directory record/\
    --init_lr 1e-4\
    --batch_size 4\
    --gpus '0 1'\
    --input_height 128\
    --input_width 256\
    --num_epochs 200\
    --train_network disnet\
    --build_network disnet\