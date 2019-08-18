CUDA_VISIBLE_DEVICES=1 python train.py --dataset_name gmo \
    --split_name train \
    --train_log_dir checkpoint_address \
    --save_interval_secs 2400 --save_summaries_secs 2400 --final_endpoint Mixed_7a


python train.py --dataset_name id \
    --split_name train \
    --train_log_dir checkpoint_id \
    --save_interval_secs 1200 --save_summaries_secs 2400 --final_endpoint Mixed_6a


python train.py --dataset_name date \
    --split_name train \
    --train_log_dir checkpoint_date \
    --save_interval_secs 1200 --save_summaries_secs 2400 --final_endpoint Mixed_6a

CUDA_VISIBLE_DEVICES=1 python train.py --dataset_name dateback \
    --split_name train \
    --train_log_dir checkpoint_dateback \
    --save_interval_secs 1200 --save_summaries_secs 2400 --final_endpoint Mixed_6a

CUDA_VISIBLE_DEVICES=1 python train.py --dataset_name names \
    --split_name train \
    --train_log_dir checkpoint_names \
    --save_interval_secs 2400 --save_summaries_secs 2400 --final_endpoint Mixed_7a

CUDA_VISIBLE_DEVICES=1 python train.py --dataset_name addback \
    --split_name train \
    --train_log_dir checkpoint_addback \
    --save_interval_secs 2400 --save_summaries_secs 2400 --final_endpoint Mixed_7a


CUDA_VISIBLE_DEVICES=0 python2 demo_inference.py --dataset_name gmo     \
--batch_size 1000     \
--image_path_pattern=address-valp --final_endpoint Mixed_7a \
--checkpoint checkpoint_address/model.ckpt-

CUDA_VISIBLE_DEVICES=0 python2 demo_inference.py --dataset_name id     \
--batch_size 20     \
--image_path_pattern=id/pad \
--checkpoint checkpoint_id/model.ckpt-

CUDA_VISIBLE_DEVICES=0 python2 demo_inference.py --dataset_name date     \
--batch_size 50     \
--image_path_pattern=test/date/pad \
--checkpoint checkpoint_date/model.ckpt-

CUDA_VISIBLE_DEVICES=0 python2 demo_inference.py --dataset_name names     \
--batch_size 20     \
--image_path_pattern=test/name/pad \
--checkpoint checkpoint_name/model.ckpt-


CUDA_VISIBLE_DEVICES=1 python2 -u demo_inference.py --dataset_name addback     \
--batch_size 500     \
--image_path_pattern=addback-valp --final_endpoint Mixed_7a \
--checkpoint checkpoint_addback/model.ckpt-

cd ..
rm -r checkpoint_momentum
CUDA_VISIBLE_DEVICES=0 python train.py --dataset_name gmo \
    --split_name train \
    --train_log_dir checkpoint_momentum \
    --save_interval_secs 1200 --save_summaries_secs 2400




CUDA_VISIBLE_DEVICES=0 python2 demo_inference.py --dataset_name dateback     \
--batch_size 100     \
--image_path_pattern=test/dateback/pad \
--checkpoint checkpoint_dateback/model.ckpt-