import streamlit as st
st.set_page_config(layout="wide") 

import SessionState

from src.sourcedoc import SourceTM, SourceDIF         
from src.fuzzy import FuzzyModels

import numpy as np

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

def run():
    
    srcs = {
        'hmds_tm':SourceTM(data_dir='data/tm_wiki/hmds'),
        'rib_tm':SourceTM(data_dir='data/tm_wiki/rib'),
        'hmds_dif':SourceDIF(data_dir='data/dif/hmds'),
        'rib_dif':SourceDIF(data_dir='data/dif/rib'),
    }
    
    fuzzymodels = FuzzyModels(
        model_name='bert-base-uncased', 
        path_dict = {
            'base':'bert-base-uncased',
            'hmds':'./outputs/lm_hmds-wiki_bert-uncased/pytorch_model.bin',
            'rib':'./outputs/lm_rib-wiki_bert-uncased/pytorch_model.bin'
        }
    )
    ###############################################
    ### Sidebar
    ###############################################
    
    st.sidebar.header('Fuzzy Entity Generator')
    
    query = st.sidebar.text_area("Enter Search Terms", )
    
    hmds_dif_base, hmds_dif_hmds, hmds_dif_rib = None, None, None
    rib_dif_base, rib_dif_hmds, rib_dif_rib = None, None, None
    hmds_tm_base, hmds_tm_hmds, hmds_tm_rib = None, None, None
    rib_tm_base, rib_tm_hmds, rib_tm_rib = None, None, None
    
    if st.sidebar.button("Search"):
        hmds_dif_base = query_search(query, fuzzymodels, 'base', srcs['hmds_dif'])
        hmds_dif_hmds = query_search(query, fuzzymodels, 'hmds', srcs['hmds_dif'])
        hmds_dif_rib = query_search(query, fuzzymodels, 'rib', srcs['hmds_dif'])
        
        rib_dif_base = query_search(query, fuzzymodels, 'base', srcs['rib_dif'])
        rib_dif_hmds = query_search(query, fuzzymodels, 'hmds', srcs['rib_dif'])
        rib_dif_rib = query_search(query, fuzzymodels, 'rib', srcs['rib_dif'])
        
        hmds_tm_base = query_search(query, fuzzymodels, 'base', srcs['hmds_tm'])
        hmds_tm_hmds = query_search(query, fuzzymodels, 'hmds', srcs['hmds_tm'])
        hmds_tm_rib = query_search(query, fuzzymodels, 'rib', srcs['hmds_tm'])
        
        rib_tm_base = query_search(query, fuzzymodels, 'base', srcs['rib_tm'])
        rib_tm_hmds = query_search(query, fuzzymodels, 'hmds', srcs['rib_tm'])
        rib_tm_rib = query_search(query, fuzzymodels, 'rib', srcs['rib_tm'])


    ###############################################
    ### Main Page
    ###############################################
    
    st.header('HMDS DIF')
    
    col1, col2, col3, col4 = st.beta_columns(4)
    col1.write('Unique Words')
    col2.write('Similar (BASE)')
    col3.write('Similar (HMDS)')
    col4.write('Similar (RIB)')
    
    col1.write(list(srcs['hmds_dif'].uniq_words))
    if hmds_dif_base:
        col2.write(format_results(hmds_dif_base))
    if hmds_dif_hmds:
        col3.write(format_results(hmds_dif_hmds))
    if hmds_dif_rib:
        col4.write(format_results(hmds_dif_rib))
    
    st.header('RIB DIF')
    
    col5, col6, col7, col8 = st.beta_columns(4)
    col5.write('Unique Words')
    col6.write('Similar (BASE)')
    col7.write('Similar (HMDS)')
    col8.write('Similar (RIB)')
    
    col5.write(list(srcs['rib_dif'].uniq_words))
    if hmds_dif_base:
        col6.write(format_results(rib_dif_base))
    if hmds_dif_hmds:
        col7.write(format_results(rib_dif_hmds))
    if hmds_dif_rib:
        col8.write(format_results(rib_dif_rib))
        
    st.header('HMDS TM')
    
    col9, col10, col11, col12 = st.beta_columns(4)
    col9.write('Unique Words')
    col10.write('Similar (BASE)')
    col11.write('Similar (HMDS)')
    col12.write('Similar (RIB)')
    
    col9.write(list(srcs['hmds_tm'].uniq_words))
    if hmds_dif_base:
        col10.write(format_results(hmds_tm_base))
    if hmds_dif_hmds:
        col11.write(format_results(hmds_tm_hmds))
    if hmds_dif_rib:
        col12.write(format_results(hmds_tm_rib))
    
    st.header('RIB TM')
    
    col13, col14, col15, col16 = st.beta_columns(4)
    col13.write('Unique Words')
    col14.write('Similar (BASE)')
    col15.write('Similar (HMDS)')
    col16.write('Similar (RIB)')
    
    col13.write(list(srcs['rib_tm'].uniq_words))
    if hmds_dif_base:
        col14.write(format_results(rib_tm_base))
    if hmds_dif_hmds:
        col15.write(format_results(rib_tm_hmds))
    if hmds_dif_rib:
        col16.write(format_results(rib_tm_rib))

   
    


if __name__ == "__main__":
    run()
