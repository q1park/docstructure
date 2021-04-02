import streamlit as st
import SessionState

import glob
import torch
from src.sourcedoc import SourceTM, SourceDIF         
# from src.fuzzy import FuzzyModels
from src.search import query_search, ranked_indices, format_results, query_search2, query_search3


import os
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

@st.cache(allow_output_mutation=True)
def load_config(model_name, **kwargs):
    return AutoConfig.from_pretrained(model_name, **kwargs)
    
@st.cache(allow_output_mutation=True)
def load_tokenizer():
    t = AutoTokenizer.from_pretrained('bert-base-uncased', TOKENIZERS_PARALLELISM=False)
    return t

@st.cache(allow_output_mutation=True)
def load_models(path_dict, config):
    return {
        k:AutoModelForMaskedLM.from_pretrained(v, config=config) 
        for k,v in path_dict.items()
    }

class FuzzyModels:
    """
    Class to store pre-trained models
    
    Attributes:
    model_name - Name of pre-trained model type
    config - Configuration dictionary for pre-trained model type
    tokenizer - Tokenizer for pre-trained model type
    models - Dictionary with key:value pairs as label:model
    """
    config_kwargs = {
        "cache_dir": None,
        "revision": "main",
        "use_auth_token": None,
        "summary_activation": "tanh",
        "summary_last_dropout": 0.1,
        "summary_type": "mean",
        "summary_use_proj": True
    }
    
    def __init__(self, model_name, path_dict):
        self.model_name = model_name
        self.path_dict = path_dict

        self.config = load_config(self.model_name, **FuzzyModels.config_kwargs)
        self.tokenizer = load_tokenizer()
        self.models = load_models(self.path_dict, self.config)

    def _embed2(self, model, data):
        return model(**data,output_hidden_states=True).hidden_states[-1].mean(dim=1).squeeze()

import re
from src.utils import uniq

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

def get_lines(df, page, elo, tlo):
    return df[(df.page==page)&(df.elo==elo)&(df.tlo==tlo)]

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

def get_by_page2(df, highlight, idxs_cols=['page', 'elo', 'tlo']):
    lil_results = []
    
    for page, elo, tlo in uniq(df[idxs_cols].to_records(index=False).tolist()):
        lil_results.append([])
        lines = get_lines(df, page, elo, tlo)
        
        for x in lines.groupby('line'):
            lil_results[-1].append(highlight(' '.join(x[1].content)))
            
        if len(lil_results[-1][0])>200:
            lil_results[-2]+=lil_results.pop(-1)
        else:
            lil_results[-1][0] = '{}-{}-{} {}'.format(tlo, elo, page, lil_results[-1][0])
    return lil_results

fuzzymodels = FuzzyModels(
        model_name='bert-base-uncased', 
        path_dict = {
            'hmds':'./outputs/lm_hmds-wiki_bert-uncased/pytorch_model.bin',
            'rib':'./outputs/lm_rib-wiki_bert-uncased/pytorch_model.bin'
        }
    )
    

def run():
    srcs = {
        'hmds':SourceDIF(data_dir='data/dif/hmds'),
        'rib':SourceDIF(data_dir='data/dif/rib'),
    }

    pdf_urls = {
        # 'hmds':glob.glob('data/tm/hmds/*pdf')[0],
        # 'rib':glob.glob('data/tm/rib/*pdf')[0]
        'hmds': 'http://insight.d2cybersecurity.com/data/tm/hmds/TM_hmds.pdf',
        'rib': 'http://insight.d2cybersecurity.com/data/tm/rib/TM_rib.pdf#page=32'
    }
    state = SessionState.get(
        full=[],
        search=[],
    )
    st.image('FutruesCommand_MAITS.png', use_column_width=True)
        
    course = st.sidebar.selectbox('Select Technical Manual:', sorted(list(srcs), reverse=True))
    state.full = get_by_page(srcs[course], Highlight([]))
    
    with st.sidebar.beta_expander('Advanced Settings', expanded=False):
        thresh = st.slider("Fuzzy Level", min_value=0, max_value=20, value=3, step=1)
        topn = st.slider("Number of Results", min_value=1, max_value=100, value=10, step=1)

       
        
    st.text('Document List:')
    st.write("   [{}-TM]({})".format(course, pdf_urls[course]))

    query = st.text_area("Enter Search Terms:", '', height=1)
        
    results = {}
    
    if st.button("Search"):   
                    
        tokenized_data = {
            q:{k:torch.LongTensor(v).unsqueeze(0) for k,v in fuzzymodels.tokenizer(q).items()}
            for q in query.split()
        }
        
        results = query_search3(tokenized_data, fuzzymodels, course, srcs[course], thresh=0, topk=thresh)
        if len(results)>0:
            with st.sidebar.beta_expander("Show Match Terms", expanded=False):
                st.write(format_results(results))

            st.header('Search Results')

            if len(results)>0:
                rankings = ranked_indices(srcs[course], results).iloc[:topn]
                highlight = Highlight([x[1] for y in results.values() for x in y])
                state.search = get_lil_results(srcs[course], rankings, highlight)

                for i, page in enumerate(state.search):
                    expanded = True if i==0 else False
                    if len(page)>1:            
                        with st.beta_expander(page[0], expanded=expanded):
                            for line in page[1:]:
                                st.write(line)
          
    with st.sidebar.beta_expander("Suggestion Box", expanded=True):
        eg = st.text_area("What You Searched:", '', height=1)
        good = st.text_area("What You Sought:", '', height=1)
        bad = st.text_area("What You Got:", '', height=1)
        
        if st.button("Save Suggestions"):
            if not eg or len(eg)==0:
                st.write('Error: You must include "What You Searched"')
            if not good or len(good)==0:
                st.write('Error: You must include "What You Sought"')
            if not bad or len(bad)==0:
                st.write('Error: You must include "What You Got"')
            if eg and good and bad and len(bad)>0 and len(good)>0 and len(eg)>0:
                with open('suggestions.txt', "a") as f:
                    f.write('|'.join([eg, good, bad])+'\n')     
        
if __name__ == "__main__":
    run()
