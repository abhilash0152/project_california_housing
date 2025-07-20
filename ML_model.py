import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error

MODEL_FILE = 'model.pkl'
PIPELINE_FILE= 'piipeline.pkl'

def build_pipeline(num_attri, cat_attri):
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scalar', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('one_hot_enc', OneHotEncoder(handle_unknown='ignore'))
    ])
    final_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attri),
        ('cat', cat_pipeline, cat_attri)
    ])
    return final_pipeline

if not os.path.exists(MODEL_FILE):
    df = pd.read_csv('housing.csv')
    df['income_catagory']=pd.cut(df['median_income'], bins=[0,1.5,3.0,4.5,6.0,np.inf], labels=[1,2,3,4,5])
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df['income_catagory']):
        train_data = df.iloc[train_index].drop('income_catagory', axis=1)
        test_data = df.iloc[test_index].drop(['income_catagory'], axis=1)
    test_data.to_csv('test_data.csv')    
    
    features = train_data.drop('median_house_value', axis=1)
    labels = train_data['median_house_value']
    
    num_attri = features.drop('ocean_proximity', axis=1).columns.tolist()
    cat_attri = ['ocean_proximity']
    
    my_pipeline = build_pipeline(num_attri, cat_attri)
    modded_features = my_pipeline.fit_transform(features)
    
    model = RandomForestRegressor()
    model.fit(modded_features, labels)
    
    joblib.dump(model, MODEL_FILE)
    joblib.dump(my_pipeline, PIPELINE_FILE)
    
    print('Model Trained Successfully...')  
else:
    model = joblib.load(MODEL_FILE)
    my_pipeline= joblib.load(PIPELINE_FILE)
    
    testing_data = pd.read_csv('test_data.csv')  
    testing_input_data = testing_data.drop('median_house_value', axis=1, errors= 'ignore')
    testing_input = my_pipeline.transform(testing_input_data)
    predictions = model.predict(testing_input)
    testing_data['predicted_value'] = predictions
    
    testing_data.to_csv('output.csv')
    print('Inference complete')
        
