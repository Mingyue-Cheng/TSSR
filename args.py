import argparse
import time
import os
import json
import pandas as pd
parser = argparse.ArgumentParser()
# dataset and dataloader args
parser.add_argument('--load_proprecessing', type=int, default=1) 
parser.add_argument('--save_proprecessing', type=int, default=0) 
parser.add_argument('--content_path', type=str, default='content')
parser.add_argument('--encoder_path', type=str, default='')
parser.add_argument('--lr', type=float, default=0.0001) 
parser.add_argument('--num_epoch', type=int, default=20)
parser.add_argument('--data_path', type=str, default='seq_cell.csv')
parser.add_argument('--save_path', type=str, default='save')
parser.add_argument('--content', type=str, default='image')
parser.add_argument('--max_len', type=int, default=10) #
parser.add_argument('--train_batch_size', type=int, default=32) 
parser.add_argument('--valid_batch_size', type=int, default=32) 
parser.add_argument('--test_batch_size', type=int, default= 32)
parser.add_argument('--d_model', type=int, default=128) 
parser.add_argument('--eval_per_steps', type=int, default=200) 
parser.add_argument('--print_step', type=int, default=100) 
parser.add_argument('--enable_res_parameter', type=int, default=1)  
parser.add_argument('--attn_heads', type=int, default=4) 
parser.add_argument('--dropout', type=float, default=0.1)  
parser.add_argument('--d_ffn', type=int, default=256) 
parser.add_argument('--bert_layers', type=int, default = 1) 
parser.add_argument('--cross_layer', type=int, default = 4) 
parser.add_argument('--le_t', type=float, default=0.2) 
args = parser.parse_args()
# other args

DATA = pd.read_csv(args.data_path, header=None)
args.user_num = len(DATA)
num_item = DATA.max().max()
del DATA
args.num_item = int(num_item)
if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)

config_file = open(args.save_path + '/args.json', 'w')
tmp = args.__dict__
json.dump(tmp, config_file, indent=1)
print(args)
config_file.close()
