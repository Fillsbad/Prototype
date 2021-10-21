# Prototype
This is a quick way of taking any number of .txt files from a folder (here about 13.5 GB), cleaning them, connecting the multi-word phrases in them and creating a W2V model from them.

## Importing libraries
```# Importing neccessary libraries
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
