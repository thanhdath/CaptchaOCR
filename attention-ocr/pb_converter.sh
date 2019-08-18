python convert_to_pb.py --split_name=train --dataset_name passport_mrz --checkpoint  checkpoint_passport_mrz/model.ckpt-281998
python freeze_graph.py --input_graph=train.pbtxt --input_checkpoint=checkpoint_passport_mrz/model.ckpt-281998 --input_binary=false --output_graph=frozen_graph.pb --output_node_names="AttentionOcr_v1/predicted_chars"

