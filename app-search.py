import streamlit as st
st.set_page_config(layout="wide") 

import SessionState

from src.sourcedoc import SourceTM, SourceDIF         
from src.fuzzy import FuzzyModels

import numpy as np
import pandas as pd
import torch

def get_top_scores(query, vocab, embeddings, thresh, topk):
    scores = (embeddings@query)/(np.linalg.norm(embeddings, axis=-1)*np.linalg.norm(query))
    return [(vocab[i], scores[i]) for i in scores.argsort()[-topk:][::-1] if scores[i]>thresh]

def query_search(query, fuzzymodels, model, src, thresh=0.75, topk=10):
    vocab = list(src.uniq_words)
    embeddings = src.embeddings[model]
    results = {}
    
    for q in query.split():
        qq = fuzzymodels._embed(fuzzymodels.models[model], q).detach().numpy()
        results[q] = [
            (x[1], x[0]) 
            for x in get_top_scores(qq, vocab, embeddings, thresh=thresh, topk=topk)
        ]
    return results

def format_results(results):
    formatted = {}
    
    for k,v in results.items():
        formatted[k] = ['{:.3f} {}'.format(x[0], x[1]) for x in v]
    return formatted

def _word_scores(results):
    word_scores = {}
    
    for k,v in results.items():
        for score, word in v:
            if word in word_scores:
                word_scores[word] +=score
            else:
                word_scores[word] = score
    return word_scores

def _get_idxs_scores(word_scores, src, shape=None):
    all_idxs, all_scores = [], []
    for k,v in word_scores.items():
        idxs = np.array([src.idx_map[idx] for idx in src.uniq_words[k]]).T
        all_idxs.append(idxs)
        all_scores.append(np.array([v]*idxs.shape[-1]))
        
    all_idxs, all_scores = np.concatenate(all_idxs, axis=-1), np.concatenate(all_scores, axis=-1)
    return all_idxs, all_scores

def _sparse_scores(word_scores, src, shape=None):
    all_idxs, all_scores = _get_idxs_scores(word_scores, src, shape=None)
    if shape is None:
        shape = tuple(x.max()+1 for x in all_idxs)
    return torch.sparse_coo_tensor(all_idxs, all_scores, shape)

def ranked_indices(src, results):
    cols = ['tlo', 'elo', 'page', 'score']
    outcols = ['page', 'elo', 'tlo', 'score']
    if len(results)==0:
        return pd.DataFrame(columns=outcols)
    
    word_scores = _word_scores(results)
    sparse_scores = _sparse_scores(word_scores, src)
    
    sparse_scores = torch.sparse.sum(sparse_scores, dim=-1)
    sparse_scores = torch.sparse.sum(sparse_scores, dim=-1)
    
    indices = pd.DataFrame(
        torch.cat([sparse_scores.indices().T, sparse_scores.values().unsqueeze(0).T], dim=-1).numpy(),
        columns=cols
    )
    for col in cols[:-1]:
        indices[col] = indices[col].astype(np.int64)
    indices = indices.sort_values(by='score', ascending=False)
    return indices[outcols]

def compute_sparse_score(src, results):
    ws = {k:_word_scores({k:v}) for k,v in results.items()}
    
    all_idxs = []
    for k,v in ws.items():
        _idx, _ = _get_idxs_scores(v, src)
        all_idxs.append(_idx)
    all_idxs = np.concatenate(all_idxs, axis=-1)
    shape = tuple(x.max()+1 for x in all_idxs)
    ss1 = {k:_sparse_scores(v, src, shape=shape) for k,v in ws.items()}
    ss2 = {k:torch.sparse.sum(v, dim=-1) for k,v in ss1.items()}
    ss3 = {k:torch.sparse.sum(v, dim=-1) for k,v in ss2.items()}
    
    scores = [x*(1/torch.max(x.values())) for x in ss3.values()]
    return torch.sparse.sum(torch.stack(scores), dim=0)

def ranked_indices2(src, results):
    cols = ['tlo', 'elo', 'page', 'score']
    outcols = ['page', 'elo', 'tlo', 'score']
    if len(results)==0:
        return pd.DataFrame(columns=outcols)

    sparse_scores = compute_sparse_score(src, results)
    
    indices = pd.DataFrame(
        torch.cat([sparse_scores.indices().T, sparse_scores.values().unsqueeze(0).T], dim=-1).numpy(),
        columns=cols
    )
    for col in cols[:-1]:
        indices[col] = indices[col].astype(np.int64)
    indices = indices.sort_values(by='score', ascending=False)
    return indices[outcols]

import re

def get_lines(df, page, elo, tlo):
    return df[(df.page==page)&(df.elo==elo)&(df.tlo==tlo)]

class Highlight:
    def __init__(self, keywords):
        self.keywords = keywords+[k.title() for k in keywords]
        self.rep = self._make_rep()
        self.pattern = self._make_pattern()
        
    def _make_rep(self):
        return dict(sorted(zip(
            list(map(re.escape, self.keywords)), 
            list(map(lambda x: '**{}**'.format(x), self.keywords))
        ), key=lambda x: len(x[0]), reverse=True))
    
    def _make_pattern(self):
        return re.compile("|".join(self.rep.keys()))
    
    def __call__(self, text):
        if len(self.rep)>0:
            return self.pattern.sub(lambda x: self.rep[re.escape(x.group(0))], text)
        else:
            return text

def get_lil_results(src, rankings, highlight):
    lil_results = []
    
    for i,row in rankings.iterrows():
        lil_results.append([])
        page, elo, tlo = int(row.page), int(row.elo), int(row.tlo)
        lines = get_lines(src.df, page, elo, tlo)
        
        for x in lines.groupby('line'):
            lil_results[-1].append(highlight(' '.join(x[1].content)))
            
        if len(lil_results[-1][0])>200:
            text = lil_results.pop(-1)
            text = text[0].split()[:3]+text
            lil_results.append(text)

        lil_results[-1][0] = '{}-{}-{} {} ({:.2f})'.format(tlo, elo, page, lil_results[-1][0].replace('*',''), row.score)
    return lil_results

def uniq(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def get_by_page(src, highlight):
    lil_results = []
    
    for page, elo, tlo in uniq(src.df[['page', 'elo', 'tlo']].to_records(index=False).tolist()):
        lil_results.append([])
        lines = get_lines(src.df, page, elo, tlo)
        
        for x in lines.groupby('line'):
            lil_results[-1].append(highlight(' '.join(x[1].content)))
            
        if len(lil_results[-1][0])>200:
            lil_results[-2]+=lil_results.pop(-1)
        else:
            lil_results[-1][0] = '{}-{}-{} {}'.format(tlo, elo, page, lil_results[-1][0])
    return lil_results

srcs = {
    'hmds':SourceDIF(data_dir='data/dif/hmds'),
    'rib':SourceDIF(data_dir='data/dif/rib'),
}

fuzzymodels = FuzzyModels(
    model_name='bert-base-uncased', 
    path_dict = {
        'hmds':'./outputs/lm_hmds-wiki_bert-uncased/pytorch_model.bin',
        'rib':'./outputs/lm_rib-wiki_bert-uncased/pytorch_model.bin'
    }
)
    
def run():
    state = SessionState.get(
        full=[],
        search=[]
    )
    ###############################################
    ### Sidebar
    ###############################################
    
    st.sidebar.header('Select Course')
    course = st.sidebar.selectbox('Courses:', sorted(list(srcs), reverse=True))
    state.full = get_by_page(srcs[course], Highlight([]))
    
    thresh = st.sidebar.slider("Fuzziness Threshold", min_value=0.25, max_value=1.0, value=0.8, step=0.025)
    topn = st.sidebar.slider("Number of Results", min_value=1, max_value=10, value=5, step=1)
    
    results = {}
    query = st.sidebar.text_area("Enter Search Terms", 'repair engine')
    if st.sidebar.button("Search"):
        results = query_search(query, fuzzymodels, course, srcs[course], thresh=thresh, topk=20)

    if len(results)>0:
        
        
        with st.sidebar.beta_expander("Show Fuzzy Entities", expanded=True):
            st.write(format_results(results))
    ###############################################
    ### Main Page
    ###############################################

    col1, col2 = st.beta_columns(2)
    
    

    col1.header('Search Results')
    
    if len(results)>0:
        rankings = ranked_indices2(srcs[course], results).iloc[:topn]
        highlight = Highlight([x[1] for y in results.values() for x in y])
        state.search = get_lil_results(srcs[course], rankings, highlight)
    
        for page in state.search:
            if len(page)>1:            
                with col1.beta_expander(page[0]):
                    for line in page[1:]:
                        st.write(line)
    
    col2.header('Full Content')
    
    for page in state.full:
        if len(page)>1:            
            with col2.beta_expander(page[0]):
                for line in page[1:]:
                    st.write(line)
                    
if __name__ == "__main__":
    run()
