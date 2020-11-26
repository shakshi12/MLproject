NB! Code for all these below mentioned steps are present in jupyter file named - 'Pre-processing_before_model_training.ipynb' 
## Pre-processing of nam and am articles:
1. Took two columns from the files (am and nam files) - 'article' and 'label'
2. Drop rows if it has null values overall
3. If the label is 'random' then convert it into 'nam' label
4. Combine am and nam articles together in one dataframe
5. Final saved uploaded file - 'cleaned_am_nam_articles.csv'

## Apply NLP for converting articles into vectors using Word2Vec
Final file for use for model training and testing - 'word2vec_features_with_labels.csv'

