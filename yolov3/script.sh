#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py \
    --dataset="tt100k" \
	--epochs=201 \
	--batch-size=32 \
	--cfg="cfg/yolov3-tt100k-tiny.cfg" \
	--img-size=416 \
	--num-workers=16

