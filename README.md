# Sthocastic Non-convex Optimization

This is the repository of the Machine learning course project, working with the

The main purpose of this projects was to explore, implement some of the most used optimization algorithms and compare them with Stocastic Cubic Regularization (SCRN) and Stocastic Cubic Regularization with Momentum (SCRN_Momentum) in a non-convex problem.

Optimizer implemented in this repository:

- SGD
- Adam
- Stocastic Cubic Regularization (SCRN)
- Stocastic Cubic Regularization with Momentum (SCRN_Momentum)

Files in this repository:

- `optimizers.py`: contains the optimizers implemented in this repository
- `models.py`: Contain the final model implemented, an enconder with 3 conv net and 3 linear layers.
- `utils.py`

## How to run the code

1. Create a conda environment with the following command:

```bash
conda create --name <env> --file requirements.txt
```

```bash
conda  activate <env>
```

2. To run one model you have the following args

- `dataset`: dataset to use, MNIST, CIFAR10 or CIFAR100.
- `conv_numbers`: number of conv layers, default 3.
- `linear_numbers`: number of linear layers, default 3.
- `hidden`: size of the linera layers, default 128.
- `epochs`: number of epochs to run the training defautl 2.
- `batch_size`: batch size, default 100.
- `lr`: learning rate, default 0.001.
- `optimizer`: optimizer to use, SGD, Adam, Sophia, SCRN or SCRN_Momentum. can be one or a list
- `activation`: activation function, default relu.
- `scheduler`: set a scheduler for the learning rate.
- `verbose`: Whether to print detailed training progress.
- `save`: save the model, default False.
- `save_path`: path to save the model, default `./models/`.
- `model_selection`: for grid search on learning rate on the optimizers required.
- `plot`: plot the loss and accuracy, default False.

### Examples

# run one optimizer over one learning rate

```bash
python main.py --dataset MNIST --optimizer SGD --epochs 10 --batch_size 100 --lr 0.001 --save True --save_path ./models/
```

```bash
python main.py --dataset CIFAR10 --optimizer Adam --epochs 10 --batch_size 100 --lr 0.001 --save True --save_path ./models/
```

## Datsets

- [MNIST](http://yann.lecun.com/exdb/mnist/)
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)

## Principal references

- [Stochastic Cubic Regularization for Non-Convex Optimization](https://arxiv.org/pdf/1902.00996.pdf)
- [Adam: A Method for Stochastic Optimization](https://arxiv.org/pdf/1412.6980.pdf)
