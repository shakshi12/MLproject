import os
import time
import pandas as pd
import numpy as np



if True:
    #On lit les ficheirs de données, et on les recopie dans un autre fichiers, mergés et avec les labels
    data_am=pd.read_csv("./../adverse/am.csv", sep="\t", index_col=False)
    data_nam=pd.read_csv("./../nonadverse/nam.csv", sep="\t", index_col=False)

    nb_am=data_am.shape[0]
    nb_nam=data_nam.shape[0]
    data_am["label"]=0
    data_nam["label"]=1

    print(data_am.head())

    print(data_am.columns)


    res=pd.concat([data_am, data_nam])
    res=res.sample(frac=1)

    print(res.shape)
    print(res.head()[["url", "label"]])


    #we save in a new file
    res.to_csv("./data/am+nam.csv", sep="\t", index_label="index")


if True:
    data=pd.read_csv("./data/am+nam.csv", sep="\t", index_col="index")
    nb_elems=data.shape[0]
    train_data=data.iloc[:int(nb_elems*0.8)]
    test_data=data.iloc[int(nb_elems*0.8):]
    print(train_data.shape)
    print(test_data.shape)
    print(train_data.columns)
    print(train_data.head)
    print(test_data.head)
    train_data.to_csv("./data/train.csv", sep="\t", header=False)
    test_data.to_csv("./data/test.csv", sep="\t", header=False)
