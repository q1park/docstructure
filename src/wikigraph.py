##############################################################
### Creates a graph structure for related Wikipedia topics
##############################################################

import os
import glob
import re
import wikipedia
from tqdm import tqdm

from src.wikiscraper import wiki_contents_redirects
from src.utils import save_pickle, load_pickle

class WikiGraph:
    def __init__(self, data_dir, nodes=[], edges=[]):
        self.data_dir = data_dir
        self.nodes = nodes
        self.edges = edges
        self._edges = {}
        self._meta = {}
        
    def _post_init(self):
        if len(set(self.nodes))!=len(self.nodes):
            raise ValueError('duplicate nodes in node list')
            
    def save(self, path):
        save_pickle(self.__dict__, path)
        
    def load(self, path):
        self.__dict__.update(load_pickle(path))

import os
import re
from collections import Counter

import spacy
nlp = spacy.load('en_core_web_sm')

re_sslash = re.compile(r'\s+(?=\/)|(?<=\/)\s+')
re_spaces = re.compile(r'\s+')
re_realword = re.compile(r'^[a-zA-Z].*[a-zA-Z]{3}.*')
re_symbol = re.compile(r'[^a-zA-Z0-9\s]')

def split_slash(text, split=['']):
    for word in re_sslash.sub('', text).split():
        _ = '' if len(split[0])==0 else ' '
        
        if re.search(r'\/', word):
            _split = word.split('/')
            split = list(map(lambda x: x+_+_split[0], split))+list(map(lambda x: x+_+_split[1], split))
        else:
            split = list(map(lambda x: x+_+word, split))
    return split

def noun_phrases(text):
    doc = nlp(text)
    return list(map(lambda x: str(x).strip(), doc.noun_chunks))

def get_noun_phrases(text_list):
    nps_flat = [
        re_spaces.sub(' ', re_symbol.sub(' ', x))
        for y in map(noun_phrases, text_list) for x in y  if re_realword.search(x)
    ]
    return list(sorted(set([x.lower() for y in map(split_slash, nps_flat) for x in y])))

def long_words(text):
    text = re_spaces.sub(' ', re_symbol.sub(' ', text))
    return [x for x in text.split() if len(x)>7]

def get_long_words(text_list):
    lws_counts = Counter([re_symbol.sub(' ', x) for y in map(long_words, text_list) for x in y])
    return list(sorted(set([w.lower() for w,n in lws_counts.most_common() if n>0])))

def get_keywords(cleaned):
    return list(sorted(set(get_noun_phrases(cleaned.content)+get_long_words(cleaned.content))))
        
def make_wikigraph(corpus, clean_fn=lambda x: x):
    data_dir = corpus.data_dir
    wiki_dir = os.path.join(corpus.data_dir, 'wiki')
    keywords = get_keywords(clean_fn(corpus.src))
    
    if not os.path.exists(wiki_dir):
        os.mkdir(wiki_dir)
        
    course_topics = {}
    wikigraph = WikiGraph(data_dir=data_dir)
    
    for seed in tqdm(keywords):
        topics = []
        
        for x in wikipedia.search(seed)[:10]:
            if len(topics)>=7:
                break
            if re.search(r'[^a-zA-Z\s]', x):
                continue
                
            ref = '/wiki/{}'.format(x.replace(' ', '_'))
            contents, redirects = wiki_contents_redirects(ref)
            
            if len(contents[contents['tags']=='h2'])>3:
                topics.append(ref)
                if ref not in wikigraph.nodes:
                    save_path = os.path.join(data_dir, '{}.csv'.format(ref.strip('/')))
                    contents.to_csv(save_path, index=False)
                    wikigraph.nodes.append(ref)
                    
                if ref not in wikigraph._edges.keys():
                    wikigraph._edges[ref] = redirects
                    
        if len(topics)>0:
            wikigraph._meta[seed] = topics[:]
    return wikigraph

def compile_wikigraph_edges(wikigraph, n_links=5):
    node_list = wikigraph.nodes[:]
    
    for i, node in enumerate(tqdm(node_list)):
        if node not in wikigraph._edges:
            print('node {} not found in _edges'.format(node))
        else:
            for ref in wikigraph._edges[node][:n_links]:
                contents, redirects = wiki_contents_redirects(ref)
                
                if len(contents[contents['tags']=='h2'])>3:
                    if ref not in wikigraph.nodes:
                        _idx = len(wikigraph.nodes)
                        save_path = os.path.join(wikigraph.data_dir, '{}.csv'.format(ref.strip('/')))
                        contents.to_csv(save_path, index=False)
                        wikigraph.nodes.append(ref)
                    else:
                        _idx = wikigraph.nodes.index(ref)
                    wikigraph.edges.append((i, _idx))
    return wikigraph