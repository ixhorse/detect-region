CUDA_VISIBLE_DEVICES=0 python train.py --backbone resnet --lr 0.005 --workers 12 --epochs 200 --batch-size 8 --gpu-ids 0 --checkname deeplab-resnet-region --eval-interval 1 --dataset tt100k --base-size 480 --crop-size 480 --use-balanced-weights
