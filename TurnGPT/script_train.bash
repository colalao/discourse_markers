train="python turngpt/train.py --batch_size 4 --language Japanese --gpus -1"

test="CUDA_VISIBLE_DEVICES=6 python turngpt_discourse_marker/test.py --language Japanese --test_type ft_one --pca"

$train
