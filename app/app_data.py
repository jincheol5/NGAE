import argparse
from ngae import GraphUtils,DataUtils

def app_data(app_number=1):
    match app_number:
        case 1:
            """
            App 1. 
            Generate train, val, test graph list dictionary and save using pickle.
            data info:
                train:
                    graph num of each type: 100 
                    node num: 20

                val:
                    graph num of each type: 3 
                    node num: 20

                test:
                    graph num of each type: 5 
                    node num: 50, 100, 1000
            """