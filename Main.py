from flask import Flask, render_template, request
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

app = Flask(__name__)
app.secret_key = 'welcome'

# Load dataset
dataset = pd.read_csv("Dataset/startup_data.csv")

# Include new labels
data = dataset[['relationships',
                'funding_rounds',
                'funding_total_usd',
                'milestones',
                'avg_participants',
                'latitude',
                'longitude',
                'city',
                'category_code',
                'status']]

# Encode categorical columns
le_status = LabelEncoder()
le_city = LabelEncoder()
le_category = LabelEncoder()

data['status'] = le_status.fit_transform(data['status'].astype(str))
data['city'] = le_city.fit_transform(data['city'].astype(str))
data['category_code'] = le_category.fit_transform(data['category_code'].astype(str))

# Separate X and Y
Y = data['status']
X = data.drop(['status'], axis=1)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train Model
rf_cls = RandomForestClassifier(
    bootstrap=False,
    max_depth=12,
    min_samples_leaf=2,
    min_samples_split=2,
    n_estimators=100
)

rf_cls.fit(X, Y)

# ================= PREDICTION ROUTE =================

@app.route('/PredictAction', methods=['POST'])
def PredictAction():
    if request.method == 'POST':
        global rf_cls, scaler, le_city, le_category

        relation = request.form['t1']
        funding = request.form['t2']
        usd = request.form['t3']
        milestone = request.form['t4']
        participant = request.form['t5']
        latitude = request.form['t6']
        longitude = request.form['t7']
        city = request.form['t8']
        category = request.form['t9']

        # Encode categorical safely
        if city in le_city.classes_:
            city_encoded = le_city.transform([city])[0]
        else:
            city_encoded = 0

        if category in le_category.classes_:
            category_encoded = le_category.transform([category])[0]
        else:
            category_encoded = 0

        data = [[
            float(relation),
            float(funding),
            float(usd),
            float(milestone),
            float(participant),
            float(latitude),
            float(longitude),
            city_encoded,
            category_encoded
        ]]

        data = scaler.transform(data)
        predict = rf_cls.predict(data)[0]

        # Prediction Result
        if predict == 0:
            result = "<font color='green'>Your startup will be Success</font>"
        else:
            result = "<font color='red'>Your startup will be Failed</font>"

        # Suggestions
        if float(milestone) < 4:
            suggestion = "Our prediction says that your start-up might failed as Milestones are less.<br/>Please try to increase milestones then you will definitely succeed."
        elif float(relation) < 4:
            suggestion = "Our prediction says that your start-up might failed as Relationships are less.<br/>Please try to increase relationships then you will definitely succeed."
        elif float(funding) < 2:
            suggestion = "Our prediction says that your start-up might failed as Funding Rounds are less."
        else:
            suggestion = "Our prediction says that your start-up will hopefully be a success."

        # Styled Output (LIKE YOUR IMAGE)
        output = f"""
<center>
<h1>Start-Up Success Prediction Screen</h1>

<font color='blue' size='4'>
Relationships = {relation} <br><br>
Funding Rounds = {funding} <br><br>
Average Funding USD = {usd} <br><br>
Milestones = {milestone} <br><br>
Average Participant = {participant} <br><br>
Latitude = {latitude} <br><br>
Longitude = {longitude} <br><br>
City = {city} <br><br>
Category Code = {category} <br><br>
</font>

<font size='4'>
<b>{result}</b>
</font>

<br><br>

<font color='blue' size='4'>
<b>Suggestion =</b> {suggestion}
</font>
</center>
"""


        return render_template('PredictSuccess.html', data=output)


# ================= OTHER ROUTES =================

@app.route('/PredictSuccess')
def PredictSuccess():
    return render_template('PredictSuccess.html', data='')

@app.route('/index')
def index():
    return render_template('index.html', data='')

@app.route('/Logout')
def Logout():
    return render_template('index.html', data='')

if __name__ == '__main__':
    app.run(debug=True)
