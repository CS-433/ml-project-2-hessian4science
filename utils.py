import pandas as pd
import numpy as np
import argparse


parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')

parser.add_argument("-dataset", default="Cifar10")
parser.add_argument("-model", default="ResNet18")
parser.add_argument("-optimizer", default="SGD")
parser.add_argument("-lr", default=0.1, type=float)
parser.add_argument("-batch_size", default=128, type=int)
parser.add_argument("-epochs", default=100, type=int)
args = parser.parse_args()
print(args.dataset, args.model, args.optimizer, args.lr, args.batch_size, args.epochs)
