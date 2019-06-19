 CUDA_VISIBLE_DEVICES=0 python trainval_net.py \
                    --dataset tt100k --net res101 \
                    --bs 2 --nw 4 --epochs 40 \
                    --lr 0.002 --lr_decay_step 30 \
                    --cuda
