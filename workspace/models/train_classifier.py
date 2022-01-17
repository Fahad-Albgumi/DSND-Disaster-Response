# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
import re
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score, recall_score, accuracy_score,  f1_score,  make_scorer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection  import GridSearchCV

def load_data(database_filepath):
    """
    Function: load data from database and return X and y.
    Args:
      database_filepath(str): database file name included path
    Return:
      X(pd.DataFrame): messages for X
      y(pd.DataFrame): labels part in messages for y
      category_names(str):category names
    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df=pd.read_sql('SELECT * from disasterETLpipeline', engine)
    X = df['message']
    # result is multiple classifty.
    Y = df.iloc[:,4:]
    category_names=Y.columns.values
    return X, Y, category_names


def tokenize(text):
    
    '''
    Tokenize text messages and cleaning for Machine Learning use.
    Input: string
    Output: list
    '''
    
     # Normalize text
    text = text.lower() 
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    
    
     # Tokenize text
    words = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    
    # Remove stop words
    tokens = [w for w in words if w not in stopwords.words('english')]  

    
    # iterate through each token
    cleanTokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip() # [WordNetLemmatizer().lemmatize(w) for w in tokens]
        cleanTokens.append(clean_tok)
    
    return cleanTokens


def build_model():
    
    '''
    Function specifies the pipeline and the grid search parameters to build a classification model by SKLEARN pipeline library 
     
    Output:  cv: classification model
    '''
    
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))
    ])
    
    parameters = parameters = {
        
        'clf__estimator__max_depth': [15, 30],  
        'clf__estimator__n_estimators': [100, 250]}
    
    

    cv = GridSearchCV(
        pipeline, 
        param_grid=parameters,
        cv=3,
        verbose=3)
        

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    Evaluate model on the test data and print the result
    input:
        model  = trained model for evaluation
        X_test = testing features (Unseen data)
        Y_test = true values to compare with prediction on unseen test cases
        category_names = column name of Y_test data
    output:
        print a panda dataframe that contains accuracy, precision, and recall, and f1 scores\
        for each output category of the dataset
    """
        
        
    y_pred = model.predict(X_test)
    
    for i,col in enumerate(category_names):
        print(col)
        #print(classification_report(np.hstack(Y_test.values.astype(int)), np.hstack(y_pred.values.astype(int))))
        print(classification_report(Y_test, y_pred, target_names=category_names))
        print("***\n")
    pass

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()