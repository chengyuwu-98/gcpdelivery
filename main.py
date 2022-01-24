from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

import pandas as pd
import wikipedia

app = Flask(__name__)


@app.route('/')
def depression_detection():
    """Return the webpage of questions form."""
    return render_template('index.html')
    # return '<h1>Welcome to the web!<h1>\
    # <img src="https://bluelightliving.com/wp-content/uploads/2020/09/Duke-University-Duke-Chapel-1-1-1080x675.jpg">'

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
    # return '<h1>Welcome to the web!<h1>\
    # <img src="https://bluelightliving.com/wp-content/uploads/2020/09/Duke-University-Duke-Chapel-1-1-1080x675.jpg">'


@app.route('/name/<value>')
def name(value):
    """Return a friendly HTTP greeting."""
    return 'Welcome to the webpage, %s!' % value

@app.route('/html')
def html():
    """Returns some custom HTML"""
    return """
    <title>This is my first web Page</title>
    <p>Hello</p>
    <p><b>World</b></p>
    """

@app.route('/pandas')
def pandas_sugar():
    df = pd.read_csv("https://raw.githubusercontent.com/noahgift/sugar/master/data/education_sugar_cdc_2003.csv")
    return jsonify(df.to_dict())


@app.route('/wikipedia/<company>')
def wikipedia_route(company):

    # Imports the Google Cloud client library
    from google.cloud import language
    result = wikipedia.summary(company, sentences=10)

    client = language.LanguageServiceClient()
    document = language.Document(
        content=result,
        type_=language.Document.Type.PLAIN_TEXT)
    encoding_type = language.EncodingType.UTF8
    entities = client.analyze_entities(request = {'document': document, 'encoding_type': encoding_type}).entities
    return str(entities)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
