from flask import Flask
from config import Config

from nltk.corpus import stopwords
from nltk import PorterStemmer
from pickle import load

app = Flask(__name__)
#app.config['SECRET_KEY'] = 'you-will-never-guess'
app.config.from_object(Config)
app.debug=True

LR_model = load(open('app/static/LogisticRegression.sav', 'rb'))
SVC_model = load(open('app/static/LinearSVC.sav', 'rb'))
DecisionTree_model = load(open('app/static/DecisionTreeClassifier.sav', 'rb'))
RF_model = load(open('app/static/RandomForestClassifier.sav', 'rb'))

stop_words = stopwords.words('english')
porter = PorterStemmer()


from app import routes