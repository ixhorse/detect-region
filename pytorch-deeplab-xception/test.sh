CUDA_VISIBLE_DEVICES=1 python test.py --backbone mobilenet --workers 12 --test-batch-size 64 --gpu-ids 0 --dataset tt100k --crop-size 480 --weight "run/tt100k/deeplab-mobile-region/experiment_33/checkpoint_122.pth.tar"
