import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import root_mean_squared_error

#1 Load the dataset

housing = pd.read_csv("housing.csv")


#2 create a stratfied split of the dataset into training and testing sets

housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
                               labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)

#we will create a copy of the training set to avoid modifying the original data
housing = strat_train_set.copy()

# 3` Separate the predictors and the labels

# housing = strat_train_set.copy()
housing_labels = housing["median_house_value"].copy()
housing.drop("median_house_value", axis=1, inplace=True)

print(housing,housing_labels)

#4. separate the numerical and categorical columns
num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attribs = ["ocean_proximity"]

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


#6. transform the data
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared)
print(housing_prepared.shape)

# 7. train a model using the prepared data

#linear regression model
lin_reg = LinearRegression() # using the linear regression model from sklearn
lin_reg.fit(housing_prepared, housing_labels) # using the training data to fit the model where housing_prepared is the input data and housing_labels is the target variable
lin_prids = lin_reg.predict(housing_prepared)
# lin_rmse = root_mean_squared_error(housing_labels, lin_prids)
# print("Linear Regression RMSE:", lin_rmse)

lin_rmse = -cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10) #using cross validation to evaluate the model with 10 folds and using the negative root mean squared error as the scoring metric
print (pd.Series(lin_rmse).describe()) # printing the cross validation scores for the linear regression model

#decision tree model
dec_reg = DecisionTreeRegressor() 
dec_reg.fit(housing_prepared, housing_labels) #using the training data to fit the model
dec_preds = dec_reg.predict(housing_prepared) #using the training data to make predictions
dec_rmses = -cross_val_score(dec_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10) #using cross validation to evaluate the model with 10 folds and using the negative root mean squared error as the scoring metric
# dec_rmse = root_mean_squared_error(housing_labels, dec_preds) #calculating the root mean squared error
# print("Decision Tree RMSE:", dec_rmses) 
print(pd.Series(dec_rmses).describe()) #using pandas to describe the cross validation scores    


#random forest model
random_forest_reg = RandomForestRegressor()
random_forest_reg.fit(housing_prepared, housing_labels)
random_forest_preds = random_forest_reg.predict(housing_prepared)
# random_forest_rmse = root_mean_squared_error(housing_labels, random_forest_preds)
# print("Random Forest RMSE:", random_forest_rmse)

random_forest_rmse = -cross_val_score(random_forest_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
print (pd.Series(random_forest_rmse).describe()) # printing the cross validation scores for the random forest model


#cross validation for random forest model
scores = cross_val_score(random_forest_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
rmse_scores = -scores
print("Random Forest RMSE scores:", rmse_scores)    



# cat_encoder = full_pipeline.named_transformers_["cat"]["onehot"]
# cat_one_hot_attribs = list(cat_encoder.get_feature_names_out(cat_attribs))
# all_attribs = num_attribs + cat_one_hot_attribs
# new_housing_prep = pd.DataFrame(housing_prepared, columns=all_attribs, index=housing.index)

# print(new_housing_prep)