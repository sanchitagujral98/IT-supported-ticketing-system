# coding: utf-8
"""
@author: sanchitagujral98
"""

from sklearn import preprocessing
import pandas as pd
import nltk
import re
from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv(r'D:\All Internships\sopra_internship\report.csv')

def labelEncoding():
    le = preprocessing.LabelEncoder()
    y_enc = le.fit_transform(df['Assignee'])
    return y_enc


#removing alphanumeric character, decimal digit, whitespace character
for i, data in df.iterrows():
    df.at[i,'text'] = re.sub(r'[^A-Za-z]', ' ', str(data))


#converting all to lower case
df = df.apply(lambda x: x.astype(str).str.lower())

#dropping all NA
df = df.dropna()
raw_text = df['text']
#stop words
stop_words = nltk.corpus.stopwords.words('english')
df['text'] = df['text'].apply(lambda x: ' '.join(term for term in x.split() if term not in set(stop_words)))
#stemming using PorterStemmer from NLTK
porter = nltk.PorterStemmer()
df['text'] = df['text'].apply(lambda x: ' '.join(porter.stem(term) for term in x.split()))

# Save cleaned and encrypted dataset back to csv without indexes
df.to_csv(r'D:\All Internships\sopra_internship\output_report.csv', 
          header =True, index=False, index_label=False)

def dataset_labels(df,X_col_index, Y_col_index):

    X = df.iloc[:, X_col_index].values
    Y = df.iloc[:, Y_col_index].values
    return X,Y

def data_split(test_ratio,state):

    X,y = dataset_labels(df,1,0)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_ratio, random_state=state)
    return  X_train, X_test, Y_train, Y_test
