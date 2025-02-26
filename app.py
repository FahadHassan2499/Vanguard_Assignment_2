from flask import  Flask,request,jsonify
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import re
import string
import mlflow
from mlflow.tracking import MlflowClient
from model.train import train

app=Flask(__name__)

train()

# Initialize MLflow client
client = MlflowClient()

@app.route("/training",methods=["POST"])
def training():
    train()   # Give the request as {"message":"training"}
    return "training Completed" 

@app.route('/best_params', methods=['GET'])
def get_best_params():
    # Get the run_id from the query parameters
    run_id = request.args.get('run_id')
    # Fetch the run details
    run = client.get_run(run_id)
        
    # Extract the best parameters from the run data
    best_params = run.data.params
    return jsonify({"run_id": run_id, "best_params": best_params})

@app.route('/prediction',methods=['POST'])
def train():
     # loading the model from the disk
    filename='./model/model.pkl'
    cvfile='./model/cv.pkl'
    clf=joblib.load(filename)
    cv=joblib.load(cvfile)
    req_txt=request.json
    txt=req_txt["message"]
    clean=re.compile('<.*?>')
    txt=re.sub(clean,'',txt)
    txt=txt.lower()
    txt=re.sub('\[.*?\]','',txt)
    txt=re.sub('[%s]'%re.escape(string.punctuation),'',txt)
    txt=re.sub('\w*\d\w*','',txt)
    txt=re.sub('[''"",,,]','',txt)
    txt=re.sub('\n','',txt)
    txt=[txt]
    txt=cv.transform(txt).toarray()
    mypred=clf.predict(txt)
    return jsonify({"predicted_value":mypred[0]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
    