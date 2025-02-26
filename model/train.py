import pandas as pd
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split as tts
import mlflow




def train():
    with mlflow.start_run():
        # reading the dataset
        data=pd.read_csv('./data/spam.csv')

        # removing the html tags
        def clean_html(text):
            clean=re.compile('<.*?>')
            cleantext=re.sub(clean,'',text)
            return cleantext
    
        # first round of cleaning
        def clean_text1(text):
            text=text.lower()
            text=re.sub('\[.*?\]','',text)
            text=re.sub('[%s]'%re.escape(string.punctuation),'',text)
            text=re.sub('\w*\d\w*','',text)
            return text

        # second round of cleaning
        def clean_text2(text):
            text=re.sub('[''"",,,]','',text)
            text=re.sub('\n','',text)
            return text
    
        cleaned_html=lambda x:clean_html(x)
        cleaned1=lambda x:clean_text1(x)
        cleaned2=lambda x:clean_text2(x)

        data['EmailText']=pd.DataFrame(data.EmailText.apply(cleaned_html))
        data['EmailText']=pd.DataFrame(data.EmailText.apply(cleaned1))
        data['EmailText']=pd.DataFrame(data.EmailText.apply(cleaned2))

        x=data.iloc[0:,1].values
        y=data.iloc[0:,0].values

        xtrain,xtest,ytrain,ytest=tts(x,y,test_size=0.20,random_state=42)

        cv = CountVectorizer()  
        xtrain = cv.fit_transform(xtrain)

        tuned_parameters = {'kernel': ['rbf','linear'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}

        classifier = GridSearchCV(svm.SVC(), tuned_parameters, n_jobs=-1)
        classifier.fit(xtrain,ytrain)

        # printing the best model
        classifier.best_params_

        # MLflow logging (parameters, metrics, model)
        mlflow.log_params(classifier.best_params_)
        mlflow.log_metric("best_accuracy", classifier.best_score_) 
        mlflow.sklearn.log_model(classifier.best_estimator_, "model")

        xtest = cv.transform(xtest)
        ypred = classifier.predict(xtest)

        # model score
        print(accuracy_score(ytest,ypred))

    # saving the model to disk
    import pickle
    pickle.dump(classifier, open('./model/model.pkl','wb'))
    pickle.dump(cv,open('./model/cv.pkl','wb'))
    
    

