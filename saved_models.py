# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 18:05:07 2019

@author: sanchitagujral98
"""
import preprocess as p
import classification_models as cm
import datetime
import pickle
import csv
#StartTime = datetime.datetime.now()

'''y_enc = p.labelEncoding()
X_ngrams = p.tokenizer()'''

X_train, X_test, Y_train, Y_test  = p.data_split(0.33,42)

Dict = {}
#Dict['build_id'] = str(StartTime)

LogisticRegression_classifier,LogisticRegression_accuracy = cm.LogisticRegression_implemenation(X_train, X_test, Y_train, Y_test)
print('Logistic Regression Accuracy: ',LogisticRegression_accuracy)
LogisticRegression_accuracy = round(LogisticRegression_accuracy*100,2)
Dict['LogisticRegression_classifier'] = LogisticRegression_accuracy
filename = 'LogisticRegression.sav'
pickle.dump(LogisticRegression_classifier, open(filename, 'wb'))

LinearSVC_classifier,LinearSVC_accuracy = cm.LinearSVC_implemenation(X_train, X_test, Y_train, Y_test)
print('Linear SVC Accuracy:',LinearSVC_accuracy)
LinearSVC_accuracy = round(LinearSVC_accuracy*100,2)
Dict['LinearSVC_classifier'] = LinearSVC_accuracy
filename = 'LinearSVC.sav'
pickle.dump(LinearSVC_classifier, open(filename, 'wb'))

SGD_classifier,SGD_accuracy = cm.SGDClassifier_implemenation(X_train, X_test, Y_train, Y_test)
print('SGD classifier accuracy: ',SGD_accuracy)
SGD_accuracy = round(SGD_accuracy*100,2)
Dict['SGD_classifier'] = SGD_accuracy
filename = 'SGDClassifier.sav'
pickle.dump(SGD_classifier, open(filename, 'wb'))

MultinomialNB_classifier,MultinomialNB_accuracy = cm.MultinomialNB_implementation(X_train, X_test, Y_train, Y_test)
print('MultinomialNB classifier: ',MultinomialNB_accuracy)
MultinomialNB_accuracy = round(MultinomialNB_accuracy*100,2)
Dict['MultinomialNB_classifier'] = MultinomialNB_accuracy
filename = 'MultinomialNBClassifier.sav'
pickle.dump(MultinomialNB_classifier, open(filename, 'wb'))

DecisionTree_classifier,DecisionTree_accuracy = cm.DecisionTree_implementation(X_train, X_test, Y_train, Y_test)
print('Decision Tree classifier:',DecisionTree_accuracy)
DecisionTree_accuracy = round(DecisionTree_accuracy*100,2)
Dict['DecisionTree_classifier'] = DecisionTree_accuracy
filename = 'DecisionTreeClassifier.sav'
pickle.dump(DecisionTree_classifier, open(filename, 'wb'))

RandomForest_classifier,RandomForest_accuracy = cm.RandomForest_implementation(X_train, X_test, Y_train, Y_test)
print('RandomForest_classifier: ',RandomForest_accuracy)
RandomForest_accuracy = round(RandomForest_accuracy*100,2)
Dict['RandomForest_classifier'] = RandomForest_accuracy
filename = 'RandomForestClassifier.sav'
pickle.dump(RandomForest_classifier, open(filename, 'wb'))


#csv_columns = ['Date','Model Name','Accuracy']
#csv_file = "models.csv"
#with open(csv_file, 'w') as csvfile:
#    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
#    print("writer created")
#    writer.writeheader()
#    print("writerheader")
#    for data in Dict:
#        print("inside for loop")
#        writer.writerow(data)


output_filename = str(datetime.datetime.now()).replace(" ","_").replace(":","_").replace(".","_")

file_headers = ['Model', 'Accuracy']
with open(output_filename+'.csv', 'w', newline='') as f:  # Just use 'w' mode in 3.x
    writer = csv.DictWriter(f, fieldnames=file_headers)
    
    writer.writeheader() 
    writer.writerow({'Model': 'Logistic Classifier', 'Accuracy':LogisticRegression_accuracy})
    writer.writerow({'Model': 'Linear SVC', 'Accuracy':LinearSVC_accuracy})
    writer.writerow({'Model': 'SGD Classifier', 'Accuracy':SGD_accuracy})
    writer.writerow({'Model': 'Multinomial Naive Bayes Classifier', 'Accuracy':MultinomialNB_accuracy})
    writer.writerow({'Model': 'Decision Tree Classifier', 'Accuracy':DecisionTree_accuracy})
    writer.writerow({'Model': 'Random Forest Classifier', 'Accuracy':RandomForest_accuracy})