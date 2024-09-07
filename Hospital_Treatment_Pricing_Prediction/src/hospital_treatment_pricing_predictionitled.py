# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler as sc
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import scipy.stats as stats

df=pd.read_csv('/content/drive/MyDrive/Package Pricing at Mission Hospital - Data Supplement.csv')

df.head()


df.isna().sum()

df.describe()

df.info()

corr=df.select_dtypes(include=['int64','float64']).corr()
corr.head()

positive_predictor=corr['TOTAL COST TO HOSPITAL '].sort_values(ascending=False)
positive_predictor

df.columns

sns.barplot(x='TYPE OF ADMSN', y='TOTAL COST TO HOSPITAL ', data= df)

sns.lineplot(x='TOTAL LENGTH OF STAY', y='TOTAL COST TO HOSPITAL ', data=df)

sns.lineplot(y='TOTAL COST TO HOSPITAL ', x='COST OF IMPLANT', data=df)

sns.barplot(y='TOTAL COST TO HOSPITAL ', x='LENGTH OF STAY - ICU', data=df)

ICU_implant=df.groupby(['TOTAL LENGTH OF STAY', 'IMPLANT '])['TOTAL COST TO HOSPITAL '].mean().reset_index()
ICU_implant.head()

sns.lineplot(x='TOTAL LENGTH OF STAY',y='TOTAL COST TO HOSPITAL ', hue='IMPLANT ',data=ICU_implant )
# as the length of ICU stay increase the cost of the to hospiatl decreases.

sns.barplot(y='TOTAL LENGTH OF STAY', x='IMPLANT ', data=Hdata)

A_W_Height=df.groupby([(df['AGE']).astype('int'), 'BODY HEIGHT','BODY WEIGHT'])['TOTAL COST TO HOSPITAL '].mean().reset_index()
A_W_Height.head()

sns.lineplot(x='BODY WEIGHT', y='TOTAL COST TO HOSPITAL ', data=A_W_Height)

sns.scatterplot(x='AGE',y='TOTAL COST TO HOSPITAL ', hue='BODY WEIGHT', data=A_W_Height )

gender_cost=df.groupby(['GENDER', 'AGE'])['TOTAL COST TO HOSPITAL '].mean().reset_index()
gender_cost.head()
sns.lineplot (x='AGE', y='TOTAL COST TO HOSPITAL ',hue='GENDER', data=gender_cost)

Marital_cost=df.groupby('MARITAL STATUS')['TOTAL COST TO HOSPITAL '].mean().reset_index()
Marital_cost.head()
sns.barplot(x='MARITAL STATUS', y='TOTAL COST TO HOSPITAL ', data=Marital_cost)

total_cost_with_past=df.groupby('PAST MEDICAL HISTORY CODE')['TOTAL COST TO HOSPITAL '].mean().reset_index()
total_cost_with_past
sns.barplot(x='PAST MEDICAL HISTORY CODE', y='TOTAL COST TO HOSPITAL ',hue='PAST MEDICAL HISTORY CODE', data=total_cost_with_past)
plt.xticks(rotation=75)

feature, target= df[['GENDER','AGE','MARITAL STATUS','PAST MEDICAL HISTORY CODE',
                       'BODY WEIGHT','BODY HEIGHT','TOTAL LENGTH OF STAY','LENGTH OF STAY - ICU','TYPE OF ADMSN', 'COST OF IMPLANT','IMPLANT ']] , df['TOTAL COST TO HOSPITAL ']

feature.dtypes

feature.isna().sum()
feature.dtypes

from sklearn.preprocessing import OneHotEncoder
OH=OneHotEncoder(sparse_output=False, drop='first')
for col in feature.select_dtypes(include='object').columns:
  encode_cols=OH.fit_transform(feature[[col]])
  encode_cols=pd.DataFrame(encode_cols, columns=OH.get_feature_names_out([col]))
  feature=feature.drop(col, axis=1)
  feature=pd.concat([feature, encode_cols], axis=1)
feature.head()

# deal with missing values
feature.isna().sum()

feature.dtypes

"""when looking at the learning curves the value of the mSE is not good also there is a big gap between the curves when the traning size ranges from 80 to 140.

"""

feature.head()

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

x_train, x_test, y_train,y_test=train_test_split(feature, target, test_size=0.2, random_state=42)
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
y_train= sc.fit_transform(y_train.values.reshape(-1,1))
y_train=y_train.ravel()
y_test=sc.transform(y_test.values.reshape(-1,1))

param_dist = {
    'n_estimators': [100, 200, 500],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [20, 25, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'max_leaf_nodes': [None, 50, 100]
}

rfc = RandomForestRegressor(random_state=42)
random_search = RandomizedSearchCV(rfc, param_distributions=param_dist, n_iter=100, cv=5,  scoring='neg_mean_squared_error',random_state=42)

random_search.fit(x_train, y_train)

best_params = random_search.best_params_
print("Best Parameters:", best_params)

best_model = random_search.best_estimator_
y_pred = best_model.predict(x_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape}")

from sklearn.linear_model import  RidgeCV, ElasticNetCV

rig_regcv=RidgeCV(alphas=[0.1, 0.5, 0.7],cv=5)
rig_regcv.fit(x_train, y_train)
yhat_1=rig_regcv.predict(x_test)

Elastic_cv=ElasticNetCV(alphas=[0.1, 0.5, 0.7],cv=5)
Elastic_cv.fit(x_train, y_train)
yhat_2=Elastic_cv.predict(x_test)

predictions=[ yhat_1,yhat_2,y_pred]
models=[ rig_regcv, Elastic_cv,best_model]

mse_list=[]
mae_list=[]
mape_list=[]

for yhat in predictions:
  mse_list.append(mean_squared_error(y_test, yhat))
  mae_list.append(mean_absolute_error(y_test, yhat))
  mape_list.append(mean_absolute_percentage_error(y_test, yhat))

for i, model in enumerate(models):
    print(f"Model: {model}")
    print(f"  MSE: {mse_list[i]}")
    print(f"  MAE: {mae_list[i]}")
    print(f"  MAPE: {mape_list[i]}")
    print("\n")

"""I tried to find out the most important features the can give is a better perdiction and lower MSE value using permutation_importance

Let's create a learning  curves and residual graphs to visualize the performance of the models.
"""

from sklearn.model_selection import learning_curve
for model in models:
  train_sizes , train_scores , validation_scores =learning_curve(model, x_train, y_train, cv=5, scoring='neg_mean_squared_error', train_sizes =np.linspace(0.01, 1,10))
  train_score_mean=-train_scores.mean(axis=1)
  validation_score_mean=-validation_scores.mean(axis=1)
  plt.plot(train_sizes, train_score_mean, label='Training error')
  plt.plot(train_sizes, validation_score_mean, label='validation error')
  plt.xlabel('Training set size')
  plt.ylabel('MSE value')
  plt.title(f'Learning Curve for {model.__class__.__name__}')
  plt.legend()
  plt.show()

from sklearn.inspection import permutation_importance
columns_name=feature.columns
x_train=pd.DataFrame(x_train, columns= columns_name)
for model in models:
        result = permutation_importance(model, x_train, y_train, n_repeats=10, random_state=42, n_jobs=2)
        sorted_idx = result.importances_mean.argsort()
        plt.barh(x_train.columns[sorted_idx], result.importances_mean[sorted_idx])
        plt.title(f'Permutation Importance for {model.__class__.__name__}')
        plt.xlabel('Permutation Importance')
        plt.show()

