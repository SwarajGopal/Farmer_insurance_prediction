import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Initialize the Flask app
app = Flask(__name__)

# Define the columns globally
columns_to_label_encode = ['sssyName.seasonName', 'isPreviousSeasonYearInSubsidy', 'firstGoiSubsidy',
                          'sssyName.stateName', 'categoryName', 'insuranceCompanyName']

# Define the label encoding function
def label_encode_transform(X):
    global columns_to_label_encode  # Access the global variable
    X = pd.DataFrame(X)
    for column in columns_to_label_encode:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
    return X

# Load the pre-trained pipeline
with open('model.pkl', 'rb') as f:
    pipeline = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input values from the form
        sssyName_year = int(request.form['sssyName_year'])
        sssyName_seasonName = request.form['sssyName_seasonName']
        sssyName_stateName = request.form['sssyName_stateName']
        sumInsured = float(request.form['sumInsured'])
        premiumRate = float(request.form['premiumRate'])
        stateShare = float(request.form['stateShare'])
        goiShare = float(request.form['goiShare'])
        isPreviousSeasonYearInSubsidy = request.form['isPreviousSeasonYearInSubsidy'] == 'True'
        firstGoiSubsidy = request.form['firstGoiSubsidy'] == 'True'
        categoryName = request.form['categoryName']
        indemnityLevel = float(request.form['indemnityLevel'])
        farmerShare = float(request.form['farmerShare'])
        stateShareValue = float(request.form['stateShareValue'])
        goiShareValue = float(request.form['goiShareValue'])
        insuranceCompanyName = request.form['insuranceCompanyName']
        PolicyTermDays = int(request.form['Policy Term (Days)'])

        # Create input array
        features = np.array([[sssyName_year, sssyName_seasonName, sssyName_stateName,
                            sumInsured, premiumRate, stateShare, goiShare,
                            isPreviousSeasonYearInSubsidy, firstGoiSubsidy, categoryName,
                            indemnityLevel, farmerShare, stateShareValue, goiShareValue, 
                            insuranceCompanyName, PolicyTermDays]])

        # Create DataFrame
        input_df = pd.DataFrame(features, columns=['sssyName.year', 'sssyName.seasonName', 'sssyName.stateName',
                                                 'sumInsured', 'premiumRate', 'stateShare', 'goiShare',
                                                 'isPreviousSeasonYearInSubsidy', 'firstGoiSubsidy', 'categoryName',
                                                 'indemnityLevel', 'farmerShare', 'stateShareValue', 'goiShareValue', 
                                                 'insuranceCompanyName', 'Policy Term (Days)'])

        # Make prediction
        prediction = pipeline.predict(input_df)
        
        return render_template('result.html', prediction_text='Predicted Farmer Share Value: {:.2f}'.format(prediction[0]))

    except Exception as e:
        # For debugging, print the error
        print(f"Error: {str(e)}")
        # Return to the home page with an error message
        return render_template('index.html', error=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)