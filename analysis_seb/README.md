***First model: text classifier***

The model is a copy from the PyTorch tutorial https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html#initiate-an-instance. It classifies betweeen nam and am the articles from the body of the article.

**Preprocessing of data**

The data considered is from the files:
- adverse/am.csv
- nonadverse/nam.csv

It does not take into account random articles.

The preprocessing is done in the file ./data_preprocessing.py: it combines the 2 previously mentioned data files into one, mixes the rows, and adds a label in order to know whether each article is am or nam. It produces the files ./data/am+nam.csv, ./data/test.csv, ./data/train.csv. Only ./data/test.csv and ./data/train.csv are used afterwards.

**Data formatting for Torch compliance**

Then, the data from ./data/train.csv and ./data/test.csv is converted to a Torch format using the content of ./data_formatting_torch.py. The file does not need to be run, it is used in main2.py. It converts the articles'bodies to vectors.

**Training the model**

In this file ./main2.py, the model TextSentiment is trained on 20 epochs, batch_size=16. At each epoch, the accuracy is measured using cross-validation. At the end, we use the test dataset to measure the accuracy of the model: it is around 80%.
