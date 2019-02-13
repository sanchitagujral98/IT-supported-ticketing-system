# coding: utf-8
"""

@author: sanchitagujral98
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

#fitting Logistic Regression to dataset
def LogisticRegression_implemenation(X_train, X_test, Y_train, Y_test):
    LogisticRegression_classifier = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])
    LogisticRegression_classifier = LogisticRegression_classifier.fit(X_train,Y_train)
    pred = LogisticRegression_classifier.predict(X_test)
    accuracy = (pred == Y_test).mean()
    return LogisticRegression_classifier, accuracy

#fitting Linear classification to dataset
def LinearSVC_implemenation(X_train, X_test, Y_train, Y_test):
    LinearSVC_classifier = Pipeline([('vect', CountVectorizer()),
                                     ('tfidf', TfidfTransformer()),
                                     ('clf-svc', LinearSVC()),])
    LinearSVC_classifier = LinearSVC_classifier.fit(X_train,Y_train)
    pred = LinearSVC_classifier.predict(X_test)
    accuracy =  (pred == Y_test).mean()
    return LinearSVC_classifier, accuracy

#fitting SGD classification to dataset
def SGDClassifier_implemenation(X_train, X_test, Y_train, Y_test):
    SGD_classifier = Pipeline([('vect', CountVectorizer()),
                                     ('tfidf', TfidfTransformer()),
                                     ('clf-sgd', SGDClassifier(loss='hinge', 
                                    penalty='l2',alpha=1e-3, max_iter=5, random_state=42)),])
    SGD_classifier = SGD_classifier.fit(X_train,Y_train)
    pred = SGD_classifier.predict(X_test)
    accuracy =  (pred == Y_test).mean()
    return SGD_classifier, accuracy

#fitting Gaussian NB classification to dataset
def MultinomialNB_implementation(X_train, X_test, Y_train, Y_test):
    NB_classifier = Pipeline([('vect', CountVectorizer()),
                              ('tfidf', TfidfTransformer()),
                              ('clf', MultinomialNB(alpha=0.1, class_prior=None, 
                                                  fit_prior=True)),])
    NB_classifier = NB_classifier.fit(X_train,Y_train)
    pred = NB_classifier.predict(X_test)
    accuracy =  (pred == Y_test).mean()
    return NB_classifier, accuracy

#fitting decision tree classification to dataset
def DecisionTree_implementation(X_train, X_test, Y_train, Y_test):
    DecisionTree_classifier = Pipeline([('vect', CountVectorizer()),
                              ('tfidf', TfidfTransformer()),
                              ('clf', DecisionTreeClassifier()),])
        
    DecisionTree_classifier = DecisionTree_classifier.fit(X_train,Y_train)
    pred = DecisionTree_classifier.predict(X_test)
    accuracy = (pred == Y_test).mean()
    return DecisionTree_classifier, accuracy

#fitting random forest classification to dataset
def RandomForest_implementation(X_train, X_test, Y_train, Y_test):
    RandomForest_classifier = Pipeline([('vect', CountVectorizer()),
                              ('tfidf', TfidfTransformer()),
                              ('clf', RandomForestClassifier()),])
    RandomForest_classifier = RandomForest_classifier.fit(X_train,Y_train)
    pred = RandomForest_classifier.predict(X_test)
    accuracy = (pred == Y_test).mean()
    return RandomForest_classifier, accuracy