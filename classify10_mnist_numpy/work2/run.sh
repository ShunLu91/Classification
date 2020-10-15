## baseline
#nohup python -u main.py --layers 3 --dimension 128 64 10 > ./logdir/baseline.log  2>&1 &
#
## layers
#nohup python -u main.py --layers 4 --dimension 128 64 64 10 > ./logdir/layer4.log  2>&1 &
#nohup python -u main.py --layers 5 --dimension 128 64 64 64 10 > ./logdir/layer5.log  2>&1 &
#nohup python -u main.py --layers 6 --dimension 128 64 64 64 64 10 > ./logdir/layer6.log  2>&1 &

# neurons
#nohup python -u main.py --layers 3 --dimension 64 32 10 > ./logdir/neuron0.log  2>&1 &
#nohup python -u main.py --layers 3 --dimension 256 128 10 > ./logdir/neuron1.log  2>&1 &
#nohup python -u main.py --layers 3 --dimension 512 256 10 > ./logdir/neuron2.log  2>&1 &

# weight_decay
#nohup python -u main.py --layers 3 --dimension 128 64 10 --weight_decay 3e-2 > ./logdir/wd0.log  2>&1 &
#nohup python -u main.py --layers 3 --dimension 128 64 10 --weight_decay 3e-3 > ./logdir/wd1.log  2>&1 &
#nohup python -u main.py --layers 3 --dimension 128 64 10 --weight_decay 3e-5 > ./logdir/wd2.log  2>&1 &

# dropout
#nohup python -u main.py --layers 3 --dimension 128 64 10 --dropout_rate 0.3 > ./logdir/dp0.log  2>&1 &
#nohup python -u main.py --layers 3 --dimension 128 64 10 --dropout_rate 0.6 > ./logdir/dp1.log  2>&1 &
#nohup python -u main.py --layers 3 --dimension 128 64 10 --dropout_rate 0.9 > ./logdir/dp2.log  2>&1 &

# noise
nohup python -u main.py --layers 3 --dimension 128 64 10 --noise_rate 0.3 > ./logdir/nr0.log  2>&1 &
nohup python -u main.py --layers 3 --dimension 128 64 10 --noise_rate 0.6 > ./logdir/nr1.log  2>&1 &
nohup python -u main.py --layers 3 --dimension 128 64 10 --noise_rate 0.9 > ./logdir/nr2.log  2>&1 &
