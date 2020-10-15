## baseline
#nohup python -u main.py --layers 3 --dimension 128 64 10 > ./logdir/baseline.log  2>&1 &
#
## layers
#nohup python -u main.py --layers 4 --dimension 128 64 64 10 > ./logdir/layer4.log  2>&1 &
#nohup python -u main.py --layers 5 --dimension 128 64 64 64 10 > ./logdir/layer5.log  2>&1 &
#nohup python -u main.py --layers 6 --dimension 128 64 64 64 64 10 > ./logdir/layer6.log  2>&1 &

# neurons
nohup python -u main.py --layers 3 --dimension 64 32 10 > ./logdir/neuron0.log  2>&1 &
nohup python -u main.py --layers 3 --dimension 256 128 10 > ./logdir/neuron1.log  2>&1 &
nohup python -u main.py --layers 3 --dimension 512 256 10 > ./logdir/neuron2.log  2>&1 &
