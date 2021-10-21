# W2V Prototype
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
##### Taking every .txt file from a folder and prints them into one file, where all the sentences are in different lines
```with open("/home/fillsbad/Jupyter/Texts/streamed.txt", 'w') as out:
    file_list = glob.glob(os.path.join(os.getcwd(), "/home/fillsbad/Jupyter/Texts/Articles", "*.txt"))
    for file_path in tqdm(file_list):
        with codecs.open(file_path, 'r', encoding = 'latin1') as f_input:
            file = f_input.read()
            tokens = nltk.sent_tokenize(file.replace('\n', ' '))
            for t in tokens:
                print(t, file = out)
```
##### Output example:
```The problem is why psychology does not seem to have so much interest in system theories and why it persists in the machine paradigm or the linear causal model as Kohler pointed out, while the ideas of systems theories have already been introduced not only in the domain of engineering and robotics, but also in biology and social sciences long before.

One reason why psychology is indifferent in the system theories would concern the objective of psychology.

If the objective of psychology is to control someoneâs behavior, so that even if a person is an organized whole that systems theory describes, then a psychologist does not need to pay attention to the total existence of a person, but only to some aspect that the psychologists want to control.

For example, in contemporary personality psychology, it is claimed that âthe Big Fiveâ factorsâopenness, conscientiousness, extroversion, agreeableness, neuroticismâ have been discovered to define human personality.

However, if personality signifies the totality of a person, it is clear that these factors are not sufficient to understand the characteristic of a personâs behaviors.

Earlier theories of personality psychology counted religious attitude, political opinion, citizenship, and aesthetic concern as factors of the personality.

These factors surely play very important roles in our personality; they are often decisive factors for our behaviors, but they are eliminated in the course of the âdevelopmentâ of personality psychology.

The reason of the elimination is that psychology does not concern itself with these factors, and does not have the intention to control these aspects of our life.

On the other hand, psychology is interested in Big Five factors and tries to intervene in these five aspects of oneâs life.

Psychology is interventionistâit tries to intervene in a personâs life and change its behavior for some purpose.

That is why psychology persists in causal explanation.
```
