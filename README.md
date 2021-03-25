# docstructure

## Quickstart
To crawl and collect wikipedia data starting from a seed topic:
```bash
python run_wikiscraper.py --start_topic Agnosticism --n_depth 2

```

To create training and testing data from a technical manual source (hmds or rib):
```bash
python run_make_wikicorpus.py --name hmds

```

To train a model based on the generated Wikipedia data:
```bash
./run_trainwiki.sh

```

To view wikipedia/TM data use nlpviewer.ipynb
Current experiments contained in corpus.ipynb
