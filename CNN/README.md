This folder contains files to implement a CNN from scratch. It is evaluated on the (1) MNIST and (2) CIFAR-10 datasets.

The respective Python files are `mnist.py` and `cifar10.py`. They are standalone and only use _Numpy_, _Matplotlib_, and _Numba_ as external dependencies. _Numba_, to be precise, is used for JIT compilation to speed up the activation functions, convolution, and pooling operations.

The data comes from [MNIST in CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv/data) and [The CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) respectively. More specifically, for the latter, the following commands sufficed to download the dataset:

```shell
wget -c https://www.cs.toronto.edu/\~kriz/cifar-10-python.tar.gz
tar -xvzf cifar-10-python.tar.gz
```

After running the scripts, the plots will be generated in `figures_mnist/` and `figures_cifar/` folders respectively.
