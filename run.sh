nohup python -u transfer.py --exp_name transfer  --gpu 4  --dropout_rate 0.5 \
 --epochs 100 --weight_decay 3e-4 --learning_rate 0.001 --batch_size 128  > transfer0.log  2>&1 &
