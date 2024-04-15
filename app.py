from flask import Flask,render_template,request
import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


app = Flask(__name__)
df=pd.read_csv('Final_data.csv')
df.drop(df.columns[df.columns.str.contains(
    'unnamed', case=False)], axis=1, inplace=True)
X=df.drop(columns='FEES')
y=df['FEES']
ohe=OneHotEncoder()
ohe.fit(X[['COUNTRY','COURSE TYPE','COURSE (SPECIALIZATION)']])
column_tran=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['COUNTRY','COURSE TYPE','COURSE (SPECIALIZATION)']),
                                    remainder='passthrough')


scores=[]
for i in range(50):
  X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=i)
  rf=RandomForestRegressor(n_estimators=200, random_state=5, max_samples=0.5, max_features=0.85, max_depth=10)
  pipe=make_pipeline(column_tran,rf)
  pipe.fit(X_train,y_train)
  y_pred=pipe.predict(X_test)
  scores.append(r2_score(y_test,y_pred))


X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=np.argmax(scores))
rf=RandomForestRegressor(n_estimators=200, random_state=5, max_samples=0.5, max_features=0.85, max_depth=10)
pipe=make_pipeline(column_tran,rf)
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
r2_score(y_test,y_pred)


@app.route('/')
def index():
    return render_template('homepage.html')
    

@app.route('/predict', methods=['POST'])
def predict():
    country= request.form.get('country')
    course_type=request.form.get('course_type')
    course_id=request.form.get('course_id')
    prediction=pipe.predict(pd.DataFrame([[country,course_type,course_id]], columns=['COUNTRY','COURSE TYPE','COURSE (SPECIALIZATION)']))
    final_price=str(prediction[0])
    return final_price


@app.route('/min/<prediction>', methods=['POST'])
def mini(prediction):
    course_id=request.form.get('course_id')
    df['FEES']=df['FEES'].astype(str)
    min =df[(df['FEES'] < prediction[0]) & (df['COURSE (SPECIALIZATION)'] == course_id)]
    min_courses_list = min.to_dict(orient='records')
    return min_courses_list
    
    
@app.route('/costpredict')
def cost():
    countries = sorted(df['COUNTRY'].unique())
    course_types = sorted(df['COURSE TYPE'].unique())
    courses = sorted(df['COURSE (SPECIALIZATION)'].unique())
    countries.insert(0,"Select Country")
    course_types.insert(0,"Select Course Type")
    courses.insert(0,"Select Course (SPECIALIZATION)")
    return render_template('index.html', countries=countries, course_types=course_types, courses=courses)



if __name__ == "__main__":
    app.run(debug=True)