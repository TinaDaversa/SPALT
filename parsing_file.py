import os
import argparse

def create_parser ():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data' ,type=str ,default='./data/ukpv/T6H12/fold0/' ,help='data path')
    parser.add_argument('--dataset_name' ,type=str ,default='ukpv' ,help='dataset name')
    parser.add_argument('--n_targets' ,type=int ,default=6 ,help='number of target time-steps')
    parser.add_argument('--targetcol', type=str, default="energy_target", help='name column target')
    parser.add_argument('--id_key', type=str, default="ss_id", help='name id key')
    parser.add_argument('--output', type=str, default="./results/ukpv/", help='path output')
    parser.add_argument('--load_model', type=str, default=None, help='path model saved')

    return parser