import argparse
import networkx as nx
import numpy as np
from ngae import DataUtils


def test(config: dict):
    match config["test_num"]:
        case 0:
            pass
        case 1:
            """
            Test 1. algo_trajectory_to_PyG_Data()
            """
            

"""
Execute Test
"""
parser=argparse.ArgumentParser()
parser.add_argument("--test_num",type=int,default=0)
args=parser.parse_args()

config={
    "test_num":args.test_num
}

test(config=config)