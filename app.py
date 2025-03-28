from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load data
df = pd.read_csv('Final_data.csv')

# Pre-processing
df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

# Model training
X = df.drop(columns='FEES')
y = df['FEES']
ohe = OneHotEncoder()
ohe.fit(X[['COUNTRY', 'COURSE TYPE', 'COURSE (SPECIALIZATION)']])
column_tran = make_column_transformer((OneHotEncoder(categories=ohe.categories_), ['COUNTRY', 'COURSE TYPE', 'COURSE (SPECIALIZATION)']),
                                      remainder='passthrough')

# Training Random Forest Regressor
rf = RandomForestRegressor(n_estimators=200, random_state=5, max_samples=0.5, max_features=0.85, max_depth=10)
pipe_rf = make_pipeline(column_tran, rf)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe_rf.fit(X_train, y_train)

# Training Linear Regression
lr = LinearRegression()
pipe_lr = make_pipeline(column_tran, lr)
pipe_lr.fit(X_train, y_train)

# Training Ridge Regression
ridge = Ridge(alpha=1.0)
pipe_ridge = make_pipeline(column_tran, ridge)
pipe_ridge.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('homepage.html')

@app.route('/predict', methods=['POST'])
def predict():
    country = request.form.get('country')
    course_type = request.form.get('course_type')
    course_id = request.form.get('course_id')
    input_data = pd.DataFrame([[country, course_type, course_id]], columns=['COUNTRY', 'COURSE TYPE', 'COURSE (SPECIALIZATION)'])
    prediction_rf = round(pipe_rf.predict(input_data)[0],2)
    final_price=str(prediction_rf)
    return final_price
    
@app.route('/predict2', methods=['POST'])
def predict2():
    country = request.form.get('country')
    course_type = request.form.get('course_type')
    course_id = request.form.get('course_id')
    input_data = pd.DataFrame([[country, course_type, course_id]], columns=['COUNTRY', 'COURSE TYPE', 'COURSE (SPECIALIZATION)'])
    prediction_lr = round(pipe_lr.predict(input_data)[0],2)
    final_price=str(prediction_lr)
    return final_price

@app.route('/predict3', methods=['POST'])
def predict3():
    country = request.form.get('country')
    course_type = request.form.get('course_type')
    course_id = request.form.get('course_id')
    input_data = pd.DataFrame([[country, course_type, course_id]], columns=['COUNTRY', 'COURSE TYPE', 'COURSE (SPECIALIZATION)'])
    prediction_ridge = round(pipe_ridge.predict(input_data)[0],2)
    final_price=str(prediction_ridge)
    return final_price
    
@app.route('/request-course')
def request_course():
    return render_template('request_course.html')

@app.route('/submit-course-request', methods=['POST'])
def submit_course_request():
    course_name = request.form['courseName']
    course_type = request.form['courseType']
    country = request.form['country']
    print(f"The user has requested: COUNTRY: {country} COURSE TYPE: {course_type} COURSE SPECIALIZATION: {course_name}")
    return "Request for new Course Submitted Successfully!"


@app.route('/min/<prediction_rf>', methods=['POST'])
def mini(prediction_rf):
    course_id=request.form.get('course_id')
    df['FEES']=df['FEES'].astype(str)
    min =df[(df['FEES'] < prediction_rf[0]) & (df['COURSE (SPECIALIZATION)'] == course_id)]
    min_courses_list = min.to_dict(orient='records')
    return min_courses_list

@app.route('/costpredict')
def cost():
    countries = sorted(df['COUNTRY'].unique())
    course_types = sorted(df['COURSE TYPE'].unique())
    courses = sorted(df['COURSE (SPECIALIZATION)'].unique())
    countries.insert(0, "Select Country")
    course_types.insert(0, "Select Course Type")
    courses.insert(0, "Select Course (SPECIALIZATION)")
    return render_template('index.html', countries=countries, course_types=course_types, courses=courses)

if __name__ == "__main__":
    app.run(debug=True)
