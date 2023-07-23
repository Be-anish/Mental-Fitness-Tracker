from flask import Flask, render_template, request,jsonify
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)
model=''
label_encoder_year=''
label_encoder_count=''
scaler=''
# Load the trained model
with open('./model.pkl', 'rb') as file:
    model = pickle.load(file)


# Load the LabelEncoder and StandardScaler objects used during training
with open('./label_encoder_count.pkl', 'rb') as file:
    label_encoder_count = pickle.load(file)
with open('./scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


predictions=[]

@app.route('/')
def home():
    return render_template('index.html',predictions=predictions)

@app.route('/predict', methods=['POST'])
def predict():
    global predictions
    # Retrieve user inputs
    country = request.form['country']
    year = int(request.form['year'])
    schizophrenia = request.form['schizophrenia']
    bipolar = request.form['bipolar']
    eating_disorders = request.form['eating_disorders']
    anxiety_disorders = request.form['anxiety_disorders']
    drug_use_disorders = request.form['drug_use_disorders']
    depressive_disorders = request.form['depressive_disorders']
    alcohol_use_disorders = request.form['alcohol_use_disorders']
    
    # Perform server-side form validation
    if not (country.strip() and 1900 <= year <= 2099):
        return jsonify({'error': 'Please enter valid values for Country and Year.'}), 400

     # Preprocess the inputs
    # Label encode the country
    country_encoded = label_encoder_count.transform([country])

    # Convert other features to float
    country_encoded=float(country_encoded)
    year_encoded = float(year)
    schizophrenia = float(schizophrenia)
    bipolar = float(bipolar)
    eating_disorders = float(eating_disorders)
    anxiety_disorders = float(anxiety_disorders)
    drug_use_disorders = float(drug_use_disorders)
    depressive_disorders = float(depressive_disorders)
    alcohol_use_disorders = float(alcohol_use_disorders)

    # Apply StandardScaler
    inputs = [[country_encoded,year_encoded, schizophrenia, bipolar, eating_disorders, anxiety_disorders,
               drug_use_disorders, depressive_disorders, alcohol_use_disorders]]
    scaled_inputs = scaler.transform(inputs)

    # Make predictions with the loaded model using the preprocessed inputs
    prediction = model.predict(scaled_inputs)

    # Store the current prediction in the predictions list
    predictions.append({'country': country, 'year': year, 'prediction': prediction[0],'schizophrenia':schizophrenia,'bipolar':bipolar,'eating_disorders':eating_disorders,
                        'anxiety_disorders':anxiety_disorders,
    'drug_use_disorders':drug_use_disorders,
    'depressive_disorders':depressive_disorders,
    'alcohol_use_disorders':alcohol_use_disorders})
    return render_template('result.html', prediction=prediction,predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
