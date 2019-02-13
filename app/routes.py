from app import app
from flask import render_template
from app.forms import ATSForm

from app import stop_words, LR_model, SVC_model, DecisionTree_model, RF_model, porter
import re
from pandas import Series


@app.route('/', methods=['GET','POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = ATSForm()
    result = []
    if form.validate_on_submit():
        ticket_str = form.user_string.data
        LR_result, SVC_result, DT_result, RF_result = Comp_result(ticket_str)
        result = [LR_result,SVC_result,DT_result,RF_result]
    return render_template('index.html', form=form, result=result)


def Comp_result(s):
    new_s = re.sub(r'[^A-Za-z]', ' ', s).lower()
    new_s = ' '.join(term for term in new_s.split() if term not in set(stop_words))
    new_s = ' '.join(porter.stem(term) for term in new_s.split())
    s2 = Series(new_s)
    LR_result = LR_model.predict(s2)
    print("Assignee through Logistic Regression: " + LR_result)

    SVC_result = SVC_model.predict(s2)
    print("Assignee through Linear SVC: " + SVC_result)

    DT_result = DecisionTree_model.predict(s2)
    print("Assignee through Decision Tree: " + DT_result)

    RF_result = RF_model.predict(s2)
    print("Assignee through Random Forest Classifier: " + RF_result)

    #SGD_model = pickle.load(open('app/static/SGDClassifier.sav', 'rb'))
    #SGD_result = SGD_model.predict(s2)
    #print("Assignee through SGD Classifier: " + SGD_result)

    #MultinomialNB_model = pickle.load(open('app/static/MultinomialNBClassifier.sav', 'rb'))
    #NB_result = MultinomialNB_model.predict(s2)
    #print("Assignee through Multinomial NB: " + NB_result)

    return (LR_result,SVC_result,DT_result,RF_result)


#Comp_result("This is a sample sentence, showing off the stop words filtration.")
