# Prototype
This is a quick way of taking any number of .txt files from a folder (here about 13.5 GB), cleaning them, connecting the multi-word phrases in them and creating a W2V model from them.

## Importing libraries
```
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import glob
import os
from tqdm import tqdm
import codecs
import re
import spacy
from collections import defaultdict
from itertools import chain
import gensim
from gensim.test.utils import datapath
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
from gensim.models.word2vec import Text8Corpus
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser 
import multiprocessing
```
## First cleaning process
##### Takes every .txt file from a folder and prints them into one file, where all the sentences are in different lines
```with open("/home/fillsbad/Jupyter/Texts/streamed.txt", 'w') as out:
    file_list = glob.glob(os.path.join(os.getcwd(), "/home/fillsbad/Jupyter/Texts/Articles", "*.txt"))
    for file_path in tqdm(file_list):
        with codecs.open(file_path, 'r', encoding = 'latin1') as f_input:
            file = f_input.read()
            tokens = nltk.sent_tokenize(file.replace('\n', ' '))
            for t in tokens:
                print(t, file = out)
```
