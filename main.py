import os
import pandas as pd
import numpy as np  
import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error

MODEL_FILE = "model.pkl "
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attribs, cat_attribs):
    #5. lets make pipeline for numerical attributes
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])
    #for categorical column, attributes we will use one hot encoding with handle_unknown='ignore' to avoid errors when we have new categories in the test set that were not present in the training set.
    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])
    # construct a full pipeline that will apply the numerical and categorical pipelines to the appropriate columns
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),         
    ])
    
    return full_pipeline

if not os.path.exists(MODEL_FILE) or not os.path.exists(PIPELINE_FILE):
    # Load the dataset
    housing = pd.read_csv("housing.csv")

    #create a stratfied split of the dataset into training and testing sets
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
                                   labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        housing.loc[test_index].drop("income_cat", axis=1).to_csv("input_data.csv", index=False)# save the test set to a csv file for inference
        housing = housing.loc[train_index].drop("income_cat", axis=1) # 
       


        housing_labels = housing["median_house_value"].copy()
        housing.features = housing.drop("median_house_value", axis=1)

        num_attribs = housing.features.drop("ocean_proximity", axis=1).columns.tolist()
        cat_attribs = ["ocean_proximity"]

        pipeline = build_pipeline(num_attribs, cat_attribs)
        housing_prepared = pipeline.fit_transform(housing.features)

        model = RandomForestRegressor(random_state=42)
        model.fit(housing_prepared, housing_labels)

        joblib.dump(model, MODEL_FILE)
        joblib.dump(pipeline, PIPELINE_FILE)    
        print ("model is trained and saved to disk, CONGRATS!!!")
else:
    #lets do inference using the saved model and pipeline
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input_data.csv")
    transformed_data = pipeline.transform(input_data)
    predictions = model.predict(transformed_data)
    input_data["median_house_value"] = predictions

    input_data.to_csv("predictions.csv", index=False)
    print ("predictions are saved to predictions.csv file, CONGRATS!!!")