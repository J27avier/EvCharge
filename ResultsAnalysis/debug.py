import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--a", help="Type of a", type=str, required=True)
args = parser.parse_args()

a = args.a
a += 1
print(a)