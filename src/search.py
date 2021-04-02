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

def query_search2(query, fuzzymodels, model, tokenizer, src, thresh=0.75, topk=10):
    vocab = list(src.uniq_words)
    embeddings = src.embeddings[model]
    results = {}

    for q in query.split():
        data = {k:torch.LongTensor(v).unsqueeze(0) for k,v in tokenizer(q).items()} 
        qq = fuzzymodels._embed2(fuzzymodels.models[model], data).detach().numpy()
        results[q] = [
            (x[1], x[0]) 
            for x in get_top_scores(qq, vocab, embeddings, thresh=thresh, topk=topk)
        ]

    return results


def query_search3(tokenized_data, fuzzymodels, model, src, thresh=0.75, topk=10):
    vocab = list(src.uniq_words)
    embeddings = src.embeddings[model]
    results = {}

    for q,data in tokenized_data.items():
        qq = fuzzymodels._embed2(fuzzymodels.models[model], data).detach().numpy()
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

def ranked_indices(src, results):
    cols = ['tlo', 'elo', 'page', 'score']
    outcols = ['page', 'elo', 'tlo', 'score']
    if len(results)==0:
        return pd.DataFrame(columns=outcols)

    sparse_scores = compute_sparse_score(src, results)
    indices = pd.DataFrame(torch.cat([
        sparse_scores.indices().T, 
        sparse_scores.values().unsqueeze(0).T
    ], dim=-1).numpy(), columns=cols)
    
    for col in cols[:-1]:
        indices[col] = indices[col].astype(np.int64)
    indices = indices.sort_values(by='score', ascending=False)
    return indices[outcols]
