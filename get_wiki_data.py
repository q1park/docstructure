import argparse
import os
import re
import random
import pandas as pd
from glob import glob

from urllib.request import urlopen
from bs4 import BeautifulSoup

re_relics = re.compile(r'\[[a-z0-9]{1,4}\]|\n')
re_spaces = re.compile(r'\s+')

def clean_text(text):
    text = re_relics.sub(' ', text)
    text = re_spaces.sub(' ', text).strip()
    return text

def convert_to_df(content):
    re_breaks = re.compile(r'Further\sreading|External\slinks|References|Navigation\smenu|Languages')
    
    data = []
    tags = []
    sections = []
    subsections = []
    section_idx = -1
    subsection_idx = 0
    
    for i, (tag, text) in enumerate(content):
        text = clean_text(text)
        if len(text)==0 or (i<len(content)-1 and content[i+1][0]==tag and tag.startswith('h')):
            continue
            
        if re_breaks.search(text):
            break
            
        if tag=='h1' or tag=='h2':
            section_idx+=1
        elif tag=='h3':
            subsection_idx+=1

        data.append(text)
        tags.append(tag)
        sections.append(section_idx)
        subsections.append(subsection_idx)
    return pd.DataFrame({'data':data, 'tags':tags, 'sections':sections, 'subsections':subsections})

def paragraphs_to_sentences(df):
    cols = ['nword', 'data', 'tag', 'section', 'subsection', 'paragraph']
    df_dict = {k:[] for k in cols}
    
    n_para = 0
    re_sentence = re.compile(r'(?<=[\.\?\!])\s')
    for i,row in df.iterrows():
        if row['tags']=='p':
            sentences = re_sentence.split(row['data'])
            n_sent = len(sentences)
            n_words = [len(x.split()) for x in sentences]
            df_dict['nword'].extend(n_words)
            df_dict['data'].extend(sentences)
            df_dict['tag'].extend(['p']*n_sent)
            df_dict['section'].extend([row['sections']]*n_sent)
            df_dict['subsection'].extend([row['subsections']]*n_sent)
            df_dict['paragraph'].extend([n_para]*n_sent)
            n_para+=1
        else:
            df_dict['nword'].append(len(row['data'].split()))
            df_dict['data'].append(row['data'])
            df_dict['tag'].append(row['tags'])
            df_dict['section'].append(row['sections'])
            df_dict['subsection'].append(row['subsections'])
            df_dict['paragraph'].append(n_para)
            n_para+=1
    return pd.DataFrame(df_dict)

class WikiScraper:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.saved_refs = self.get_saved_refs()
        self.dataframes = {}
        
        self.base_url = 'https://en.wikipedia.org'
        self.tags = ['h1', 'h2', 'h3', 'p']
        
    def get_saved_refs(self):
        return [re.search(r'\/wiki/[^\.]+', x).group() for x in glob(self.data_dir+'/*')]
        
    def get_wiki_data(self, ref):
        def get_tag_text(elem):
            return elem.name, elem.text
        
        re_links = re.compile(r'^\/wiki\/[a-zA-Z]+$')
        source = urlopen(self.base_url+ref).read()# Make a soup 
        soup = BeautifulSoup(source,'lxml')
        found_list = list(self.dataframes.keys())

        content = list(map(get_tag_text, soup.find_all(lambda x: x.name in self.tags)))
        redirects = [x['href'] for x in soup.find_all(href=True) if re_links.search(x['href'])]
        
        content = convert_to_df(content)
        redirects = [x for x in redirects if x not in found_list+self.saved_refs]
        return content, redirects
    
    def collect_wiki_data(self, refs, max_num):
        if len(self.dataframes)>max_num:
            return None
        
        refs = refs[:max(int(len(refs)/2), 1)]
        random.shuffle(refs)
        for ref in refs:
            df, redirects = self.get_wiki_data(ref)
            if len(df.subsections.unique())>5:
                print('saving: {}'.format(ref))
                self.dataframes[ref] = paragraphs_to_sentences(df)
                
                if len(self.dataframes)>max_num:
                    return None
                else:
                    self.collect_wiki_data(redirects, max_num)
                
            else:
                print('skipping {} ..too short'.format(ref))
            
    def save_wiki_data(self):
        for k,v in self.dataframes.items():
            v.to_csv('data'+k+'.csv', index=False)
            
        self.dataframes = {}
        self.tooshort = {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_topic",
        type=str,
        required=True,
        help="Article name of starting point for crawl",
    )

    parser.add_argument(
        "--n_depth",
        default=2,
        type=int,
        help="Number of recursions in crawl",
    )
    
    args = parser.parse_args()
    
    scraper = WikiScraper('data/wiki')
    scraper.collect_wiki_data(['/wiki/{}'.format(args.start_topic)], max_num=args.n_depth)
    scraper.save_wiki_data()


if __name__ == "__main__":
    main()