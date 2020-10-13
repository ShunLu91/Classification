#nohup python -u classify10_mnist_numpy/mlp_mnist.py > ./logdir/baseline.log  2>&1 &
#nohup python -u classify10_mnist_numpy/mlp_mnist.py > ./logdir/layer2.log  2>&1 &
#nohup python -u classify10_mnist_numpy/mlp_mnist.py > ./logdir/layer4.log  2>&1 &
#nohup python -u classify10_mnist_numpy/mlp_mnist.py > ./logdir/neurons_l.log  2>&1 &
#nohup python -u classify10_mnist_numpy/mlp_mnist.py > ./logdir/neurons_s.log  2>&1 &
nohup python -u classify10_mnist_numpy/mlp_mnist.py > ./logdir/wd_l.log  2>&1 &
