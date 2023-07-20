import tensorflow as tf
#from flask_ngrok import run_with_ngrok
from tensorflow import keras
import numpy as np
from tensorflow.keras.layers import TextVectorization
from flask import Flask, request, render_template,redirect,url_for
#from flask import Flask
import pickle

app = Flask(__name__)
#run_with_ngrok(app)

model=keras.models.load_model("cmodel.h5")
from_disk = pickle.load(open("tv_layer.pkl", "rb"))
new_vectorizer = TextVectorization(max_tokens=from_disk['config']['max_tokens'],
                                          output_mode=from_disk['config']['output_mode'],
                                          output_sequence_length=from_disk['config']['output_sequence_length'])
new_vectorizer.set_weights(from_disk['weights'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Pass/<int:score>')
def Pass(score):
    return " the result is pass and marks are " +str(score)

@app.route('/fail/<int:score>')
def fail(score):
    return "the result is fail and marks are " +str(score)

@app.route('/getprediction',methods=['GET','POST'])
def getprediction():
    input =request.form.get('comment')    
    final_input = new_vectorizer(input)
    prediction = model.predict(np.expand_dims(final_input,0))
   # prediction=np.round(prediction,decimals=5)
    classe=["Toxic","Severe_toxic","Obscene","Threat","Insult","Identity_hate"]
    label1=np.argmax(prediction)
   # temp1=prediction[0,label1]
    pr1=np.round(prediction[0,label1],decimals=4)
    prediction[0,label1]=0
    label2=np.argmax(prediction)
    #temp2=prediction[0,label2]
    pr2=np.round(prediction[0,label2],decimals=4)
    if pr1<0.5:
         return render_template('index.html',output ="the comment is Non-Toxic")
    else:

        return render_template('index.html',output ="the comment toxicity level is {0} ({1}) as well as {2} ({3}) ".format(classe[label1],pr1,classe[label2],pr2))

@app.route('/Result/<int:mks>')
def Results(mks):
    res=""
    if mks>50:
        res="Pass"
    else:
        res="fail"
    return redirect(url_for(res,score=mks))

if __name__ == '__main__':
    app.run()	