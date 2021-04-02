##############################################################
### Pretrained-model management
##############################################################

import os
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

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

        self.config = AutoConfig.from_pretrained(self.model_name, **FuzzyModels.config_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=None, use_fast=True)
        self.models = {
            k:AutoModelForMaskedLM.from_pretrained(v, config=self.config) 
            for k,v in path_dict.items()
        }
    
    def _embed(self, model, text):
        return model(
            **{k:torch.LongTensor(v).unsqueeze(0) for k,v in self.tokenizer(text).items()},
            output_hidden_states=True
        ).hidden_states[-1].mean(dim=1).squeeze()
                
    def _embed2(self, model, data):
        return model(**data,output_hidden_states=True).hidden_states[-1].mean(dim=1).squeeze()
    
#     def embed(self, model, texts):
#         _embeddings = []

#         for text in tqdm(texts):
#             _embeddings.append(self._embed(model, text))
#         return torch.stack(_embeddings).detach().numpy()
    