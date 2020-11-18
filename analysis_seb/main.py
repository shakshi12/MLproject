from nltk.tokenize import word_tokenize
from nltk import pos_tag, RegexpParser
from sklearn.feature_extraction.text import CountVectorizer
import csv
import pandas as pd


data_am=pd.read_csv("./../adverse/am.csv", sep="\t", index_col=False)
data_nam=pd.read_csv("./../nonadverse/nam.csv", sep="\t", index_col=False)

#print(data_am.head())
#print(data_nam.head())
print(data_am.columns)
print(data_nam.columns)

text=data_am.iloc[0]['article'].split()

if False:
    tokens_tag = pos_tag(text)
    print("After Token:",tokens_tag)
    patterns= """mychunk:{<NN.?>*<VBD.?>*<JJ.?>*<CC>?}"""
    chunker = RegexpParser(patterns)
    print("After Regex:",chunker)
    output = chunker.parse(tokens_tag)
    print("After Chunking",output)
    output.draw()

if False:
    vectorizer=CountVectorizer()
    vocabulary=vectorizer.fit(text)
    X= vectorizer.transform(text)
    print(X.toarray())
    print(vocabulary.get_feature_names())
