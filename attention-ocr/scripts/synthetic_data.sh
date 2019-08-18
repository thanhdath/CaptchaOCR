cd data_synthesize
python gen_dateback.py --n_sample 100000
cd ..
CUDA_VISIBLE_DEVICES=1 python train.py --dataset_name dateback \
    --split_name train \
    --train_log_dir checkpoint_dateback \
    --save_interval_secs 1200 --save_summaries_secs 2400 --final_endpoint Mixed_6a

CUDA_VISIBLE_DEVICES=0 python2 demo_inference.py --dataset_name dateback     \
--batch_size 50     \
--image_path_pattern=test/date/pad \
--checkpoint checkpoint_dateback/model.ckpt-

