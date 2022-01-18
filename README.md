# Disaster Response Pipeline Project

## Introduction

After a catastrophic event, a great many individuals convey messages to request help through different channels like web-based media. Like that they need food; or they're caught under rubble. However, the government does not have ample time to read all the messages and send them to related departments. Then, at that point, this task will assume a significant part and assist individuals and keep them safe.
This Project is needed as a piece of the Data Science Nanodegree Program of Udacity in a joint effort with appen. The underlying dataset contains pre-marked tweet and messages from genuine disaster situations. The point of the project is to assemble a Natural Language Processing instrument that classifies messages.
## Requirements

python (=>3.6)
pandas
numpy
sqlalchemy
sys
plotly
sklearn
joblib
flask
nltk


## File Structure
app
| - template
| |- master.html # main page of web app
| |- go.html # classification result page of web app
|- run.py # Flask file that runs app
data
|- disaster_categories.csv # data to process
|- disaster_messages.csv # data to process
|- process_data.py
|- InsertDatabaseName.db # database to save clean data to
models
|- train_classifier.py
|- classifier.pkl # saved model
README.md
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
