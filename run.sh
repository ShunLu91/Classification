nohup python -u transfer.py --exp_name transfer1 --model_name resnet50 --gpu 3  --dropout_rate 0.5 --cutout \
 --epochs 100 --weight_decay 3e-4 --learning_rate 0.001 --batch_size 128  > log1  2>&1 &
