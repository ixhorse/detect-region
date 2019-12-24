 CUDA_VISIBLE_DEVICES=0 python trainval_net.py \
                    --dataset tt100k --net res101 \
                    --bs 4 --nw 4 --epochs 60 \
                    --lr 0.0001 --lr_decay_step 57 \
                    --cuda \
                    --r True\
                    --checksession 1 \
                    --checkepoch 55 \
                    --checkpoint 15170
