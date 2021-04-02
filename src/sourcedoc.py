##############################################################
### Create language-model training data from a source document
##############################################################

import re
import numpy as np
import pandas as pd

from src.utils import load_numpy, save_numpy, load_pickle, save_pickle

re_spaces = re.compile(r'\s+')
re_symbols = re.compile(r'[^a-zA-Z0-9\s\']')

def clean_word(text):
    text = re_symbols.sub(' ', text)
    text = re_spaces.sub(' ', text).strip()
    return text

re_sentence = re.compile(r'(?<=[\.\?\!])\s')

def splitby_sentence(text):
    return re_sentence.split(text)

def splitby_word(text):
    return text.split()

def split_data(df, split_fn, split_name, split_col='data'):
    assert split_col in df.columns
    
    df_dict = {split_name:[]}
    df_dict.update({k:[] for k in df.columns})
    
    for i, split in enumerate(map(split_fn, df[split_col])):
        for k in df_dict.keys():
            if k==split_col:
                df_dict[k].extend(split)
            elif k==split_name:
                df_dict[k].extend([i]*len(split))
            else:
                df_dict[k].extend([df[k].iloc[i]]*len(split))
    return pd.DataFrame(df_dict)

def reindex(idxs_list, sparse=False):
    n_depth = len(idxs_list[0])
    out_idxs = []
    _idx = [0]*n_depth
    for j, idx in enumerate(idxs_list):
        if j>0:
            for i in range(n_depth):
                if i<n_depth-1 and idx[i+1:]!=idxs_list[j-1][i+1:]:
                    if sparse:
                        _idx[i] = 0
                    else:
                        _idx[i] +=1
                elif idx[i]!=idxs_list[j-1][i]:
                    _idx[i] +=1
        out_idxs.append(tuple(_idx))
    return out_idxs

def reindex_df(df, idxs_cols, sparse=True):
    og_cols = df.columns.tolist()
    assert all(x in og_cols for x in idxs_cols)
    
    idxs = df[idxs_cols].to_records(index=False).tolist()
    idxs = reindex(idxs, sparse=sparse)
    return pd.concat([
        df.drop(columns=idxs_cols),
        pd.DataFrame(idxs, columns=idxs_cols)
    ], axis=1)[og_cols]

import os
import glob
import pandas as pd

class SourceTM:
    def __init__(self, data_dir, idx_labels=['word', 'group', 'subsubsection', 'subsection', 'section']):
        if not os.path.exists(os.path.join(data_dir, 'corpus.csv')):
            raise ValueError('No file corpus.csv in {}'.format(data_dir))
        self.data_dir = data_dir
        self.idx_labels = idx_labels
        self.src  = pd.read_csv(os.path.join(data_dir, 'corpus.csv')).dropna()
        
        self.df = self._load_df()
        self.idx_map = self._make_index_map()
        self.uniq_words = self._make_uniq_words()
        self.embeddings = self._load_embeddings()
        
    def _make_index_map(self):
        idx_map_path = os.path.join(self.data_dir, 'idx_map.pkl')
        if os.path.exists(idx_map_path):
            return load_pickle(idx_map_path)
        else:
            idx_map = {
                k:tuple(reversed(v)) 
                for k,v in zip(self.df.index.tolist(), self.df[self.idx_labels].to_records(index=False))
            }
            save_pickle(idx_map, idx_map_path)
            return idx_map
    
    def _make_uniq_words(self):
        uwords_path = os.path.join(self.data_dir, 'uwords.pkl')
        
        if os.path.exists(uwords_path):
            return load_pickle(uwords_path)
        else:
            self.df['uncased'] = self.df.content.apply(lambda x: clean_word(x).lower())
            uniq_words = dict(
                map(
                    lambda x: (x[0], sorted(x[1].index.tolist())), 
                    self.df.sort_values(by='uncased').groupby('uncased')
                )
            )
            save_pickle(uniq_words, uwords_path)
            return uniq_words

    def _load_df(self):
        df_path = os.path.join(self.data_dir, 'df.csv')
        if os.path.exists(df_path):
            df = pd.read_csv(df_path).dropna()
        else:
            df = self._split_reindex()
            df.to_csv(df_path, index=False)
        return df
    
    def _split_reindex(self):
        df = split_data(self.src, splitby_word, 'group', split_col='content')
        df.reset_index(inplace=True)
        df.rename(columns = {'index':'word'}, inplace=True)
        return reindex_df(df, self.idx_labels)
    
    def _load_embeddings(self):
        return {
            re.search(r'[^\/]+(?=\_emb\.npy)', emb_path).group():load_numpy(emb_path) 
            for emb_path in glob.glob(os.path.join(self.data_dir, '*_emb.npy'))
        }
            
    def add_embeddings(self, fuzzymodels):
        for k,v in fuzzymodels.models.items():
            emb_path = os.path.join(self.data_dir, '{}_emb.npy'.format(k))
            
            self.embeddings.update({k:fuzzymodels.embed(v, list(self.uniq_words.keys()))})
            save_numpy(self.embeddings[k], emb_path)
    
import numpy as np
import glob

def merge_data(struct, src):
    cols = ['page', 'elo', 'tlo', 'type', 'content']
    merge_dict = {k:[] for k in cols}
    
    for i,row in struct.iterrows():
        text = src[src.PageID==row.pid]
        
        if len(text)==0:
            merge_dict['page'].append(row.page)
            merge_dict['elo'].append(row.elo)
            merge_dict['tlo'].append(row.tlo)
            merge_dict['type'].append(row['type'])
            merge_dict['content'].append(row['name'])
        else:
            for ii,rrow in text.iterrows():
                merge_dict['page'].append(row.page)
                merge_dict['elo'].append(row.elo)
                merge_dict['tlo'].append(row.tlo)
                merge_dict['type'].append(row['type'])
                merge_dict['content'].append(rrow.TextDisplayValue)
    return pd.DataFrame(merge_dict)

class SourceDIF:
    def __init__(self, data_dir, idx_labels=['word', 'line', 'page', 'elo', 'tlo']):
        self.data_dir = data_dir
        self.idx_labels = idx_labels
        
        self.src  = pd.read_csv(os.path.join(data_dir, 'CourseText.csv')).dropna()
        self.meta  = pd.read_csv(os.path.join(data_dir, 'CourseTextExtendedData.csv')).dropna()
        self.struct = pd.read_csv(os.path.join(data_dir, 'structure.csv')).fillna(-1)
        self.struct.pid = self.struct.pid.astype(np.int64)
        
        self.df = self._load_df()
        self.idx_map = self._make_index_map()
        self.uniq_words = self._make_uniq_words()
        self.embeddings = self._load_embeddings()

    def _make_index_map(self):
        idx_map_path = os.path.join(self.data_dir, 'idx_map.pkl')
        if os.path.exists(idx_map_path):
            return load_pickle(idx_map_path)
        else:
            idx_map = {
                k:tuple(reversed(v)) 
                for k,v in zip(self.df.index.tolist(), self.df[self.idx_labels].to_records(index=False))
            }
            save_pickle(idx_map, idx_map_path)
            return idx_map
    
    def _make_uniq_words(self):
        uwords_path = os.path.join(self.data_dir, 'uwords.pkl')
        
        if os.path.exists(uwords_path):
            return load_pickle(uwords_path)
        else:
            self.df['uncased'] = self.df.content.apply(lambda x: clean_word(x).lower())
            uniq_words = dict(
                map(
                    lambda x: (x[0], sorted(x[1].index.tolist())), 
                    self.df.sort_values(by='uncased').groupby('uncased')
                )
            )
            save_pickle(uniq_words, uwords_path)
            return uniq_words
    
    def _load_df(self):
        df_path = os.path.join(self.data_dir, 'df.csv')
        if os.path.exists(df_path):
            df = pd.read_csv(df_path).dropna()
        else:
            df = self._split_reindex(merge_data(self.struct, self.src))
            df.to_csv(df_path, index=False)
        return df
    
    def _split_reindex(self, df):
        df = split_data(df, splitby_word, 'line', split_col='content')
        df.reset_index(inplace=True)
        df.rename(columns = {'index':'word'}, inplace=True)
        return reindex_df(df, self.idx_labels)
    
    def _load_embeddings(self):
        return {
            re.search(r'[^\/]+(?=\_emb\.npy)', emb_path).group():load_numpy(emb_path) 
            for emb_path in glob.glob(os.path.join(self.data_dir, '*_emb.npy'))
        }
            
    def add_embeddings(self, fuzzymodels):
        for k,v in fuzzymodels.models.items():
            emb_path = os.path.join(self.data_dir, '{}_emb.npy'.format(k))
            
            self.embeddings.update({k:fuzzymodels.embed(v, list(self.uniq_words.keys()))})
            save_numpy(self.embeddings[k], emb_path)