##############################################################
### Random crawl through Wikipedia starting from a seed topic
##############################################################

import re
import random
import pandas as pd
from glob import glob

from urllib.request import urlopen
from bs4 import BeautifulSoup

re_relics = re.compile(r'\[[a-z0-9]{1,4}\]|\n')
re_spaces = re.compile(r'\s+')
re_links = re.compile(r'^\/wiki\/[a-zA-Z\_]+$')
re_breaks = re.compile(
    r'Further\sreading|External\slinks|References|'
    'Navigation\smenu|Languages|Disambiguation\spages'
)

def content_to_df(content):
    data, tags = [], []
    sections, subsections = [], []
    section_idx, subsection_idx = -1, 0
    prev_tag = 'h1'
    
    for i, (tag, text) in enumerate(content):
        text = re_relics.sub(' ', text)
        text = re_spaces.sub(' ', text).strip()
        
        if len(text)==0 or (i<len(content)-1 and content[i+1][0]==tag and tag.startswith('h')):
            continue
        if tag=='li':
            continue
            
        if tag=='h1' or tag=='h2':
            section_idx+=1
            subsection_idx+=1
        elif tag=='h3' and prev_tag=='p':
            subsection_idx+=1

        data.append(text)
        tags.append(tag)
        sections.append(section_idx)
        subsections.append(subsection_idx)
        prev_tag = tag
        
    df = pd.DataFrame({'subsection':subsections, 'section':sections, 'data':data, 'tags':tags})
    return df

def wiki_contents_redirects(ref):
    wiki_root = 'https://en.wikipedia.org'
    tags = ['h1', 'h2', 'h3', 'p', 'li']
    
    if len(ref)>0 and ref[0].isalpha():
        ref = '/wiki/'+ref
    soup = BeautifulSoup(urlopen('{}{}'.format(wiki_root, ref)).read(), 'lxml')

    contents, redirects = [], []
    is_toc, tag_toc = False, None
    
    for c in soup.find_all(lambda x: x.name in tags):
        if re_breaks.search(c.text) and c.name!='li':
            break
            
        if is_toc and c.name==tag_toc:
            is_toc, tag_toc = False, None
        if c.name[0].startswith('h') and c.text=='Contents':
            is_toc, tag_toc = True, c.name
        if is_toc:
            continue
        
        contents.append((c.name, c.text))
    
    for r in soup.find_all(href=True):
        if re.search(r'Main[\_\s]Page', r['href']):
            break
            
        if re_links.search(r['href']) and not r['href']==ref:
            redirects.append(r['href'])
    return content_to_df(contents), redirects

class WikiScraper:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.saved_refs = []#self.get_saved_refs()
        self.dataframes = {}
        
    def get_saved_refs(self):
        return [re.search(r'\/wiki/[^\.]+', x).group() for x in glob(self.data_dir+'/*')]
        
    def get_wiki_data(self, ref):
        contents, redirects = wiki_contents_redirects(ref)
        found_list = list(self.dataframes.keys())
        redirects = [x for x in redirects if x not in found_list+self.saved_refs]
        return contents, redirects
    
    def collect_wiki_data(self, refs, max_num):
        if len(self.dataframes)>max_num:
            return None
        
        refs = refs[:max(int(len(refs)/2), 1)]
        random.shuffle(refs)
        for ref in refs:
            df, redirects = self.get_wiki_data(ref)
            if len(df['subsection'].unique())>5:
                print('saving: {}'.format(ref))
                self.dataframes[ref] = df[['subsection', 'section', 'data', 'tags']]
                
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