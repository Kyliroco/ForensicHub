# CUDA_VISIBLE_DEVICES=0 \
# torchrun \
#   --standalone \
#   --nnodes=1 \
#   --nproc_per_node=1 \
#   training_scripts/train.py \
#   --config statics/doc/ffdn_train.yaml

# CUDA_VISIBLE_DEVICES=0,1 \
# torchrun \
#   --standalone \
#   --nnodes=1 \
#   --nproc_per_node=2 \
#   training_scripts/train.py \
#   --config statics/mask2label/trufor_train.yaml

CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  training_scripts/train.py \
  --config statics/mask2label/mesorch_train_pillow.yaml

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# torchrun \
#   --standalone \
#   --nnodes=1 \
#   --nproc_per_node=4 \
#   training_scripts/train.py \
#   --config statics/doc/adcd_net_train.yaml