#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses, datasets
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import os
import gzip
import csv
import nltk
import torch


# In[2]:


torch.cuda.set_device(1)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# If GPU is avaliable and what kind of GPU we use

print(torch.cuda.get_device_name(0))
print(torch.cuda.current_device())


# In[3]:


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Training parameters
#model_name = 'sentence-transformers/nli-bert-base-cls-pooling'
model_name = 'bert-base-uncased'
train_batch_size = 8
num_epochs = 1
max_seq_length = 75


# In[4]:


tokens=[]
with open('added.txt','r', encoding='utf8') as f:
    for line in f:
        tokens.append(line.strip('\n'))


# In[5]:


# Save path to store our model
model_save_path = 'training_stsb_tsdae-{}-{}-{}-{}_raw_vocab'.format(model_name,num_epochs, train_batch_size, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

# Check if dataset exsist. If not, download and extract  it
sts_dataset_path = 'data/stsbenchmark.tsv.gz'

if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)

# Defining our sentence transformer model
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
# add speicial token


word_embedding_model.tokenizer.add_tokens(tokens, special_tokens=True)
word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))

pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), 'cls')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# In[ ]:





# In[6]:


# We use 1 Million sentences from Wikipedia to train our model
wikipedia_dataset_path = 'training.txt'
# wikipedia_dataset_path = 'data/wiki1m_for_simcse.txt'
#if not os.path.exists(wikipedia_dataset_path):
#    util.http_get('https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt', wikipedia_dataset_path)

# train_samples is a list of InputExample objects where we pass the same sentence twice to texts, i.e. texts=[sent, sent]
train_sentences = []
fIn = open(wikipedia_dataset_path, 'r', encoding='utf8').read().split('.')
for line in fIn:
    line = line.strip()
    if len(line) >= 10:
            train_sentences.append(line)        


# In[7]:


# Read STSbenchmark dataset and use it as development set
logging.info("Read STSbenchmark dev dataset")
dev_samples = []
test_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1

        if row['split'] == 'dev':
            dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
        elif row['split'] == 'test':
            test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='sts-test')

# We train our model using the MultipleNegativesRankingLoss
train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)
train_loss = losses.DenoisingAutoEncoderLoss(model, decoder_name_or_path=model_name, tie_encoder_decoder=True)


evaluation_steps = 1000
logging.info("Training sentences: {}".format(len(train_sentences)))
logging.info("Performance before training")
dev_evaluator(model)


# In[ ]:


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=evaluation_steps,
          output_path=model_save_path,
          weight_decay=0,
          warmup_steps=100,
          optimizer_params={'lr': 3e-5},
          use_amp=True         #Set to True, if your GPU supports FP16 cores
          )
##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################


# In[ ]:


model = SentenceTransformer(model_save_path)
test_evaluator(model, output_path=model_save_path)


# In[ ]:


model_save_path


# In[ ]:




