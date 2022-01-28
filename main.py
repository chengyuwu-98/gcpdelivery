from flask import Flask
#from flask import jsonify
from flask import render_template
from flask import request

#from sklearn.preprocessing import StandardScaler
#from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle
#import pandas as pd


app = Flask(__name__)


@app.route('/')
def depression_detection():
    """Return the webpage of questions form."""
    return render_template('index.html')
   

@app.route('/predict', methods=['POST', 'GET'])
def depression_prediction():
    """Return the prediction result of depression."""
    ### Load model
    scaler_file = open("scaler.pickle", "rb")
    scaler = pickle.load(scaler_file)
    model_file = open("model.pickle", "rb")
    model = pickle.load(model_file)
    print(request.args)
    ### Load input values
    param_list = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q12', 'Q13', 'Q14', 'TIPI1', 'TIPI2', 'TIPI3', 'TIPI4', 'TIPI5', 'TIPI6', 'TIPI7', 'TIPI8', 'TIPI9', 'TIPI10', 'Education', 'Urban', 'Gender', 'Engnat', 'Age', 'Hand', 'Religion', 'Orientation', 'Race', 'Voted', 'Married', 'Familysize']
    x_list = []
    for param in param_list:
        x_list.append(float(request.args.get(param)))
    x_list = np.array(x_list)
    x_list = np.reshape(x_list, [1, -1])
    x_scale = scaler.transform(x_list)
    depression_level = model.predict(x_scale)
    return "Your depression level is : %s" %depression_level[0]
 


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
