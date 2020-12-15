import torch
from torchtext.datasets import text_classification
import pandas as pd
import numpy as np
import pickle
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from torch.utils.data.dataset import random_split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open('vocab.pkl', 'rb') as f1:
    vocab = pickle.load(f1)


import logging
import torch
import io
from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import Vocab
from tqdm import tqdm
from  torchtext.datasets.text_classification import TextClassificationDataset


NGRAMS=2

def generate_batch(batch):
    indexes=torch.tensor([int(entry[0]) for entry in batch])
    label = torch.tensor([int(entry[1]) for entry in batch])
    text = [entry[2] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    text=text.type(torch.LongTensor)
    return indexes, text, offsets, label






class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()


    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)
    
    
    def create_iterator_from_file_for_testing(self, vocab, ngrams, data, include_unk=False):
        #each element contains text and label
        article_index=1
        label_index=2
        datas = []
        labels = []
        tokenizer = get_tokenizer("basic_english")

        with tqdm(unit_scale=0, unit='lines') as t:
            for index, row in data.iterrows():
                tokens = ' '.join([str(row['article'])])
                tokens=tokenizer(tokens)
                if 'label' in row: #for compatibility, will not be used, and is not necessary
                    cls=int(row['label'])
                else:
                    cls=1
                index=row.name

                if include_unk:
                    tokens = torch.tensor([vocab[token] for token in tokens])
                else:
                    token_ids = list(filter(lambda x: x is not Vocab.UNK, [vocab[token] for token in tokens]))
                    tokens = torch.tensor(token_ids)
                if len(tokens) == 0:
                    logging.info('Row contains no tokens.')
                datas.append((index, cls, tokens))
                labels.append(cls)
                t.update(1)
        return datas, set(labels)
    
    
    
    def setup_datasets_testing_from_df(self, vocab, df, include_unk=False):
        test_data, test_labels=self.create_iterator_from_file_for_testing(vocab, NGRAMS, df)
        return TextClassificationDataset(vocab, test_data, test_labels), vocab
    
    
    
    def predict(self, df, vocab):
        test_dataset, vocab=self.setup_datasets_testing_from_df(vocab, df)
        all_indexes=[]
        predictions=[]
        data = DataLoader(test_dataset, batch_size=df.shape[0], shuffle=False, collate_fn=generate_batch)
        
        for indexes, text, offsets, cls in data:
            text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
            with torch.no_grad():
                #print(indexes, cls)
                #print(text, offsets)
                output = self.forward(text, offsets)
                
                for i in range(output.shape[0]):
                    predictions.append(np.argmax(output[i].numpy()))
                    all_indexes.append(indexes[i])
                
        return predictions


N_EPOCHS = 20
BATCH_SIZE = 16
VOCAB_SIZE = len(vocab)
EMBED_DIM = 32
NUN_CLASS = 2



with open('model.pkl', 'rb') as f1:
    model = pickle.load(f1)


#returns a list of the predicted labels
predictions=model.predict(pd.read_csv("./data/test_2.csv", delimiter=","), vocab)


#and next we get the true labels for comparison
labels=pd.read_csv("./data/test_2.csv", delimiter=",")['label']
labels

# 1-Accuracy
error=0
for i in range(len(predictions)):
    if predictions[i]!=labels[i]:
        error+=1
print(error/len(predictions))

