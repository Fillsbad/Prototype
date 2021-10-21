# W2V Prototype
This is a quick way of taking any number of .txt files from a folder (here about 13.5 GB), cleaning them, connecting the multi-word expressions in them based on a number of books (here about 5.4 GB) and creating a W2V model from them.

It is important to note that all subparts of this process are printed into separate files, so they can be checked individually. To show how ceratain parts look like, I included an example output after many blocks of code.

## Importing libraries (I'll check later if all of these are indeed neccessary)
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
#### Taking every .txt file from a folder and printing them into one file, where all the sentences are in different lines
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
```
The problem is why psychology does not seem to have so much interest in system theories and why it persists in the machine paradigm or the linear causal model as Kohler pointed out, while the ideas of systems theories have already been introduced not only in the domain of engineering and robotics, but also in biology and social sciences long before.

One reason why psychology is indifferent in the system theories would concern the objective of psychology.

If the objective of psychology is to control someoneâs behavior, so that even if a person is an organized whole that systems theory describes, then a psychologist does not need to pay attention to the total existence of a person, but only to some aspect that the psychologists want to control.

For example, in contemporary personality psychology, it is claimed that âthe Big Fiveâ factorsâopenness, conscientiousness, extroversion, agreeableness, neuroticismâ have been discovered to define human personality.

However, if personality signifies the totality of a person, it is clear that these factors are not sufficient to understand the characteristic of a personâs behaviors.

Earlier theories of personality psychology counted religious attitude, political opinion, citizenship, and aesthetic concern as factors of the personality.
```
#### Removing all characters from a text, except a-z, A-Z, &Δ*öüóőúűáéäí-
```filepath = '/home/fillsbad/Jupyter/Texts/streamed.txt'
outfile = open('/home/fillsbad/Jupyter/Texts/cleaned_az.txt', 'w')

with open(filepath) as fp:
    for line in tqdm(fp):
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        tok_sentences = tokenizer.tokenize(line)
        def sentence_cleaner(sent):
            clean = re.sub("[^a-zA-Z&Δ*öüóőúűáéäí-]"," ", sent)
            words = re.sub(' +', ' ', clean)
            return words
        clean_sentences = []
        for cleanable_sentence in tok_sentences:
            if len(cleanable_sentence) > 0:
                clean_sentences.append(sentence_cleaner(cleanable_sentence))
        print(' '.join(map(str, clean_sentences)), file = outfile)
```
##### Output example:
```The problem is why psychology does not seem to have so much interest in system theories and why it persists in the machine paradigm or the linear causal model as Kohler pointed out while the ideas of systems theories have already been introduced not only in the domain of engineering and robotics but also in biology and social sciences long before 

One reason why psychology is indifferent in the system theories would concern the objective of psychology 

If the objective of psychology is to control someone s behavior so that even if a person is an organized whole that systems theory describes then a psychologist does not need to pay attention to the total existence of a person but only to some aspect that the psychologists want to control 

For example in contemporary personality psychology it is claimed that the Big Five factors openness conscientiousness extroversion agreeableness neuroticism have been discovered to define human personality 

However if personality signifies the totality of a person it is clear that these factors are not sufficient to understand the characteristic of a person s behaviors 

Earlier theories of personality psychology counted religious attitude political opinion citizenship and aesthetic concern as factors of the personality
```
#### A little extra cleaning and putting a '.' at the end of every line/sentence
```with open('/home/fillsbad/Jupyter/Texts/cleaned_sci_th.txt', 'w') as out:
    with open('/home/fillsbad/Jupyter/Texts/cleaned_az.txt') as f:
        for li in tqdm(f):
            line = li.replace(' rst ',' ')
            line = li.replace(' st ',' ')
            line = li.replace(' nd ',' ')
            line = li.replace(' rd ',' ')
            line = li.replace(' th ',' ')
            line = li.replace(' s ',' ')
            line = li.replace(' \n','.\n')
            print(line.strip(), file = out)
```
##### Output example:
```The problem is why psychology does not seem to have so much interest in system theories and why it persists in the machine paradigm or the linear causal model as Kohler pointed out while the ideas of systems theories have already been introduced not only in the domain of engineering and robotics but also in biology and social sciences long before.

One reason why psychology is indifferent in the system theories would concern the objective of psychology.

If the objective of psychology is to control someone s behavior so that even if a person is an organized whole that systems theory describes then a psychologist does not need to pay attention to the total existence of a person but only to some aspect that the psychologists want to control.

For example in contemporary personality psychology it is claimed that the Big Five factors openness conscientiousness extroversion agreeableness neuroticism have been discovered to define human personality.

However if personality signifies the totality of a person it is clear that these factors are not sufficient to understand the characteristic of a person s behaviors.

Earlier theories of personality psychology counted religious attitude political opinion citizenship and aesthetic concern as factors of the personality.
```
## Connecting multi-word expressions in the text, based on the psychology books collected
In this part I will not include the output examples as the cleaning process is the same as before
#### (BOOK CLEANING) Taking every .txt file from a folder and printing them into one file, where all the sentences are in different lines
```with open("/home/fillsbad/Jupyter/Texts/Training/streamed.txt", 'w') as out:
    file_list = glob.glob(os.path.join(os.getcwd(), "/home/fillsbad/Jupyter/Texts/Books", "*.txt"))
    for file_path in tqdm(file_list):
        with codecs.open(file_path, 'r', encoding = 'latin1') as f_input:
            file = f_input.read()
            tokens = nltk.sent_tokenize(file.replace('\n', ' '))
            for t in tokens:
                print(t.strip(), file = out)
 ```               
#### (BOOK CLEANING) Removing all characters from a text, except a-z, A-Z, &Δ*öüóőúűáéäí-
```filepath = '/home/fillsbad/Jupyter/Texts/streamed.txt'
outfile = open('/home/fillsbad/Jupyter/Texts/cleaned_az.txt', 'w')

with open(filepath) as fp:
    for line in tqdm(fp):
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        tok_sentences = tokenizer.tokenize(line)
        def sentence_cleaner(sent):
            clean = re.sub("[^a-zA-Z&Δ*öüóőúűáéäí-]"," ", sent)
            words = re.sub(' +', ' ', clean)
            return words
        clean_sentences = []
        for cleanable_sentence in tok_sentences:
            if len(cleanable_sentence) > 0:
                clean_sentences.append(sentence_cleaner(cleanable_sentence))
        print(' '.join(map(str, clean_sentences)), file = outfile)
```
#### (BOOK CLEANING) A little extra cleaning and putting a '.' at the end of every line/sentence
```with open('/home/fillsbad/Jupyter/Texts/cleaned_sci_th.txt', 'w') as out:
    with open('/home/fillsbad/Jupyter/Texts/cleaned_az.txt') as f:
        for li in tqdm(f):
            line = li.replace(' rst ',' ')
            line = li.replace(' st ',' ')
            line = li.replace(' nd ',' ')
            line = li.replace(' rd ',' ')
            line = li.replace(' th ',' ')
            line = li.replace(' s ',' ')
            line = li.replace(' \n','.\n')
            print(line.strip(), file = out)
```
#### Tokenizing the training text - in this case, a collection of 4500+ psychology books - so we can train the gensim Phrases model
```with open('/home/fillsbad/Jupyter/Texts/Training/cleaned_books_th.txt') as inf, open('/home/fillsbad/Jupyter/Texts/Training/processed_books.txt', 'w') as out:
    for line in tqdm(inf):
        line = nltk.sent_tokenize(line)
        print(line, file = out)
```
##### Output example:
```['Three parts follow each of these x Preface rst-person accounts']

['The chapters in each section are written by authorities selected for their knowledge in the eld of military psychology sociology and other social sciences and shed light on the reality of life in the armed forces']

['This set integrates the diverse in uences on the well-being and performance of military personnel by developing separate volumes that address different facets of military psychology']

['By focusing on Military Performance the rst volume addresses the need to understand the determinants of how military personnel think react and behave on military operations']

['Several of the chapters in Volume also have implications for the well-being of military personnel such as the consequences of killing how stress affects decision making and how sleep loss affects operational effectiveness']

['Newly emerging issues in the armed forces are also discussed including the role of terrorism psychological operations and advances in optimizing cognition on the battle eld'
```
#### Creating a Phrases model based on the training corpus, then freezing and saving it (by doing this process twice as seen below, we can connect not only the bigrams but also the 3 and sometimes the 4 word expressions as well)
```sentences = Text8Corpus('/home/fillsbad/Jupyter/Texts/Training/processed_books.txt')
bigram = Phrases(sentences, min_count=1, threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS)
trigram = Phrases(bigram[sentences], min_count=1, threshold=1, connector_words=ENGLISH_CONNECTOR_WORDS)
frozen_model = trigram.freeze()
frozen_model.save('/home/fillsbad/Jupyter/Texts/Training/frozen3.pkl')
```
#### Connecting the phrases with an underscore based on the frozen model we trained on the Books corpus
```with open('/home/fillsbad/Jupyter/Texts/connected.txt', 'w') as out:
    model_reloaded = Phrases.load('/home/fillsbad/Jupyter/Texts/Training/frozen3.pkl')
    sentences = Text8Corpus('/home/fillsbad/Jupyter/Texts/cleaned_sci_th.txt')
    for lines in tqdm(sentences):
        print(model_reloaded[lines], file = out)
```
#### Transforming the text back into normal sentences - this is potentially the worst way to do this
```with open('/home/fillsbad/Jupyter/Texts/connected.txt') as inf, open('/home/fillsbad/Jupyter/Texts/normal_again.txt', 'w') as out:
    for line in tqdm(inf):
        line = line.replace('[','')
        line = line.replace("'",'')
        line = line.replace(']','')
        line = line.replace(',','')
        line = nltk.sent_tokenize(line)
        for l in line:
            print(l, file = out)
```
##### Output example:
```The problem is why psychology does_not seem to have so_much interest in system theories and why it persists in the machine paradigm or the linear_causal model as Kohler pointed_out while the ideas of systems theories have_already been_introduced not_only in the domain of engineering and robotics but_also in biology and social_sciences long before.

One_reason why psychology is indifferent in the system theories would concern the objective of psychology.

If the objective of psychology is to control someone s behavior so that even_if a person is an organized whole that systems theory describes then a psychologist does_not need to pay_attention to the total existence of a person but only to some_aspect that the psychologists want to control.

For_example in contemporary personality psychology it is claimed_that the Big_Five factors openness_conscientiousness extroversion_agreeableness neuroticism have_been discovered to define human personality.

However if personality signifies the totality of a person it is clear that these factors are not_sufficient to understand the characteristic of a person s behaviors.

Earlier theories of personality psychology counted religious attitude political_opinion citizenship and aesthetic concern as factors of the personality.
```
## Checking if all the multi-word expressions we need to connect are indeed connected
This part is unfinished, in the probable case, that not every words we we want to connect are connected, we will have to connect the rest ourselves (this is a short and relatively easy process)
## Finishing the cleaning process and transforming the text, so the W2V model can be created
```filepath = '/home/fillsbad/Jupyter/Texts/normal_again.txt'
outfile = open('/home/fillsbad/Jupyter/Texts/W2V/no_sw.txt', 'w')

with open(filepath) as fp:
    for line in tqdm(fp):
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        raw_sentences = tokenizer.tokenize(line)               

        stop_words = set(stopwords.words('english')) 
        def sentence_to_wordlist(raw):
            words = ' '.join(w for w in raw.split() if w not in stop_words)
            words = words.replace('.','')
            words = words.lower()
            return words
        no_sw_sentences = []
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                no_sw_sentences.append(sentence_to_wordlist(raw_sentence))
        print(no_sw_sentences, file = outfile)
```
##### Output example:
```['the problem psychology does_not seem so_much interest system theories persists machine paradigm linear_causal model kohler pointed_out ideas systems theories have_already been_introduced not_only domain engineering robotics but_also biology social_sciences long before']

['one_reason psychology indifferent system theories would concern objective psychology']

['if objective psychology control someone behavior even_if person organized whole systems theory describes psychologist does_not need pay_attention total existence person some_aspect psychologists want control']

['for_example contemporary personality psychology claimed_that big_five factors openness_conscientiousness extroversion_agreeableness neuroticism have_been discovered define human personality']

['however personality signifies totality person clear factors not_sufficient understand characteristic person behaviors']

['earlier theories personality psychology counted religious attitude political_opinion citizenship aesthetic concern factors personality']
```
## Training the W2V model (fine tunining the model is unfinished)
```
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in tqdm(open(os.path.join(self.dirname, fname))):
                yield line.split()                                
tokenized = MySentences('/home/fillsbad/Jupyter/Texts/W2V')

model = Word2Vec(tokenized, min_count = 5, vector_size = 300, window = 7, workers = 4, sg = 1)
```
#### Checking the number of items in the model vocabulary - optional
```
print(len(model.wv))
```
#### Saving the model as a .bin file
```
model.wv.save_word2vec_format('/home/fillsbad/Jupyter/Texts/model3.bin', binary = True)
```
#### Loading a model later
```
loaded_model = gensim.models.KeyedVectors.load_word2vec_format('/home/fillsbad/Jupyter/Texts/model3.bin', binary = True)
```
#### Checking the closest 40 expressions to the word happiness
```
loaded_model.most_similar(['happiness'], topn = 40)
```
##### Output example:
```[("happiness']", 0.7802721858024597),
 ("['happiness", 0.7136465907096863),
 ('happi_ness', 0.7061863541603088),
 ('happiness_and_well-being', 0.6933473944664001),
 ('happiness_happiness', 0.6896741390228271),
 ('levels_of_happiness', 0.6875686645507812),
 ('happiness_and_satisfaction', 0.6841875314712524),
 ('happiness_satisfaction', 0.6649748682975769),
 ('overall_happiness', 0.6637284755706787),
 ('subjective_well-being', 0.6610868573188782),
 ('happiness_diener', 0.6535085439682007),
 ('contentment_happiness', 0.6515305042266846),
 ('subjective_wellbeing', 0.6508346796035767),
 ("['world_database", 0.645182728767395),
 ('pursuing_happiness', 0.6451472043991089),
 ('unhappiness', 0.6449677348136902),
 ('life-satisfaction', 0.6441795825958252),
 ('hap_piness', 0.6411221027374268),
 ('contentment', 0.6408823132514954),
 ('life_satisfaction', 0.6402313709259033),
 ('swb', 0.6388760209083557),
 ('happiness_joy', 0.6366546750068665),
 ('priority_in_public', 0.6363426446914673),
 ('eudaimonia', 0.636114776134491),
 ('eudamonic', 0.6352128386497498),
 ('satisfaction_and_contentment', 0.6310755014419556),
 ('kesebir_and_diener', 0.6306400895118713),
 ('eudaemonia', 0.6298112273216248),
 ('happiness_and_contentment', 0.629772961139679),
 ('greater_happiness', 0.6233652830123901),
 ('hedonic_and_eudemonic', 0.6232475638389587),
 ('happiness_and_wellbeing', 0.622951865196228),
 ('hedonic_well-being', 0.6216959357261658),
 ('happiness_kahneman', 0.6205111742019653),
 ('hedonic_happiness', 0.6204162836074829),
 ('non-happiness', 0.6182316541671753),
 ('satisfaction_diener', 0.6178640723228455),
 ('veenhoven', 0.6172904372215271),
 ("contentment']", 0.617216944694519),
 ('chekola', 0.6160106658935547)]
 ```
