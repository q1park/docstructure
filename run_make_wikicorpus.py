import argparse
import os
import pandas as pd

from src.sourcedoc import SourceTM
from src.wikigraph import WikiGraph, make_wikigraph, compile_wikigraph_edges

import random

def clean_hmds(df):
    df.content = df.content.str.replace(
        r'[^a-zA-Z\s]{4,100}|Figure\s[^a-zA-Z]+|REF\s[^a-zA-Z]+|VMDS|'
        '[A-Z]+\-[A-Z]+|[^\s]*[a-zA-Z][0-9][^\s]*|[^\s]*[0-9][a-zA-Z][^\s]*', '', regex=True
    )
    df.content = df.content.str.replace(r'^\s+|\s+$', '', regex=True)
    df.content = df.content.str.replace(r'^[0-9][^\s]*\s(?=[a-zA-Z])', '', regex=True)
    df = df[df.content.str.contains(r'[a-zA-Z]{3}', regex=True)].reset_index(drop=True)
    return df

def clean_rib(df):
    df.content = df.content.str.replace(
        r'[^a-zA-Z\s]{4,100}|Figure\s[^a-zA-Z]+|RIB|'
        '[A-Z]+\-[A-Z]+|[^\s]*[a-zA-Z][0-9][^\s]*|[^\s]*[0-9][a-zA-Z][^\s]*', '', regex=True
    )
    df.content = df.content.str.replace(r'^\s+|\s+$', '', regex=True)
    df.content = df.content.str.replace(r'^[0-9][^\s]*\s(?=[a-zA-Z])', '', regex=True)
    df = df[df.content.str.contains(r'[a-zA-Z]{3}', regex=True)].reset_index(drop=True)
    return df

def train_test_split(n, r_train, seed=0):
    random.seed(seed)
    idxs = list(range(n))
    random.shuffle(idxs)
    
    n_train = int(r_train*n)
    return sorted(idxs[:n_train]), sorted(idxs[n_train:])

def get_paragraph_text(path):
    df = pd.read_csv(path)
    s = df[df.tags=='p'].data.apply(lambda x: '{}\n'.format(x))
    return ''.join(s)

def get_paragraph_texts(paths):
    return ''.join(list(map(get_paragraph_text, paths)))

def make_mlm_dataset(wgraph):
    data_dir = wgraph.data_dir
    nodes = wgraph.nodes
    train_path = os.path.join(data_dir, 'train.txt')
    test_path = os.path.join(data_dir, 'test.txt')
    
    train_idxs, test_idxs = train_test_split(len(nodes), 0.9)
    train_paths = [os.path.join(data_dir, '{}.csv'.format(nodes[i].strip('/'))) for i in train_idxs]
    test_paths = [os.path.join(data_dir, '{}.csv'.format(nodes[i].strip('/'))) for i in test_idxs]
    train_data, test_data = get_paragraph_texts(train_paths), get_paragraph_texts(test_paths)
    
    with open(train_path, 'w') as f:
        f.write(train_data)
        
    with open(test_path, 'w') as f:
        f.write(test_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of TM",
    )
    
    args = parser.parse_args()
    
    data_dir = 'data/tm_wiki/{}'.format(args.name)
    save_path = os.path.join(data_dir, 'wikigraph.pkl'.format(args.name))
    
    if args.name=='hmds':
        clean_fn = clean_hmds
    elif args.name=='rib':
        clean_fn = clean_rib
    else:
        raise KeyError('{} not found'.format(args.name))
        
    if os.path.exists(save_path):
        wikigraph = WikiGraph(data_dir=data_dir)
        wikigraph.load(save_path)
    else:
        wikigraph = make_wikigraph(SourceTM(data_dir), clean_fn=clean_fn)
        wikigraph.save(save_path)
        
    wikigraph = compile_wikigraph_edges(wikigraph)
    wikigraph.save(save_path)
    make_mlm_dataset(wikigraph)

if __name__ == "__main__":
    main()