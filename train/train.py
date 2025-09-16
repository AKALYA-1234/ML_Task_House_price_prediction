#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
df=pd.read_csv(r"C:\Users\DELL\Downloads\kc_house_data.csv",encoding='latin1')


# In[2]:


df.head()


# In[3]:


df.shape


# In[4]:


df.columns


# In[5]:


df = df.drop(columns=["id", "date", "zipcode"])


# In[6]:


df.isnull().sum()


# In[7]:


df.duplicated().sum()


# In[8]:


df=df.drop_duplicates()


# In[9]:


df.duplicated().sum()


# In[10]:


df_sample = df.sample(n=5000, random_state=42) 


# In[11]:


df_sample.head()


# In[12]:


df_sample.shape


# In[13]:


df_sample['waterfront'].unique()


# In[14]:


df_sample['view'].unique()


# In[15]:


df_sample['condition'].unique()


# In[16]:


df_sample['grade'].unique()


# In[ ]:





# In[17]:


df_sample.skew()


# In[18]:


import numpy as np
import pandas as pd

df_before = df_sample.copy()
df_after = df_sample.copy()

log_cols = ['price','sqft_living','sqft_lot','sqft_above',
            'sqft_living15','sqft_lot15']

for col in log_cols:
    df_after[col] = np.log1p(df_after[col])

skew_before = df_before[log_cols].skew()
skew_after = df_after[log_cols].skew()

skew_comparison = pd.DataFrame({
    'Before_Log': skew_before,
    'After_Log': skew_after
})

print("Skewness comparison before vs after log transformation")
print(skew_comparison)


# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns
for col in log_cols:
    plt.figure(figsize=(12,5))
    
    plt.subplot(1,2,1)
    sns.histplot(df_before[col], bins=40, kde=True, color="skyblue")
    plt.title(f"Before Log Transformation: {col}")
    
    plt.subplot(1,2,2)
    sns.histplot(df_after[col], bins=40, kde=True, color="orange")
    plt.title(f"After Log Transformation: {col}")
    
    plt.tight_layout()
    plt.savefig("plots/distrbutions.png")
    plt.close()


# In[20]:


plt.figure(figsize=(14,10))
corr = df_sample.corr()
sns.heatmap(corr, annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap of Features", fontsize=16)
plt.savefig("plots/Heatmap.png")
plt.close()


# In[21]:


plt.figure(figsize=(12,6))
sns.scatterplot(x='long', y='lat', hue='price', data=df_sample,
                palette="viridis", alpha=0.5)
plt.title("House Prices by Geographic Location (lat, long)")
plt.savefig("plots/House price by lat and long.png")
plt.close()


# In[22]:


plt.figure(figsize=(10,5))
sns.barplot(x='bedrooms', y='price', data=df_sample, estimator=np.mean,
            errorbar=None, hue='bedrooms', legend=False,palette="dark:blue")
plt.title("Average Price vs Bedrooms")
plt.savefig("plots/Price Vs BedroomS.png")
plt.close()


# In[23]:


plt.figure(figsize=(10,5))
sns.barplot(x='bathrooms', y='price', data=df_sample, estimator=np.mean,
            errorbar=None, hue='bathrooms', legend=False,palette="dark:Green")
plt.title("Average Price vs Bathrooms")
plt.savefig("plots/Price Vs Bathroom.png")
plt.close()


# In[24]:


plt.figure(figsize=(10,5))
sns.barplot(x='floors', y='price', data=df_sample, estimator=np.mean,
            errorbar=None, hue='floors', legend=False,palette="dark:Orange")
plt.title("Average Price vs Floors")
plt.savefig("plots/Price Vs Floors.png")
plt.close()


# In[25]:


plt.figure(figsize=(10,5))
sns.barplot(x='condition', y='price', data=df_sample, estimator=np.mean,
            errorbar=None, hue='condition', legend=False,palette="dark:purple")
plt.title("Average Price vs Condition")
plt.savefig("plots/Price Vs Condition.png")
plt.close()


# In[26]:


plt.figure(figsize=(10,5))
sns.barplot(x='grade', y='price', data=df_sample, estimator=np.mean,
            errorbar=None, hue='grade', legend=False, palette="dark:red")
plt.title("Average Price vs Grade")
plt.savefig("plots/Price Vs Grade.png")
plt.close()


# In[27]:


plt.figure(figsize=(10,5))
sns.barplot(x='view', y='price', data=df_sample, estimator=np.mean,
            errorbar=None, hue='view', legend=False, palette="coolwarm")
plt.title("Average Price vs View")
plt.savefig("plots/Price Vs View.png")
plt.close()

# In[28]:


plt.figure(figsize=(6,5))
sns.barplot(x='waterfront', y='price', data=df_sample, estimator=np.mean,
            errorbar=None, hue='waterfront', legend=False, palette="Set2")
plt.title("Average Price: Waterfront vs Non-Waterfront")
plt.savefig("plots/Waterfront Vs Non waterfront.png")
plt.close()


# In[29]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
X = df.drop("price", axis=1) 
y = df["price"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0, max_iter=10000, random_state=42)
    ),
    "Lasso Regression": make_pipeline(
        StandardScaler(),
        Lasso(alpha=0.1, max_iter=20000, random_state=42)   
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, max_depth=5, min_samples_split=10, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
    ),
    "AdaBoost": AdaBoostRegressor(
        n_estimators=100, learning_rate=0.05, random_state=42
    ),
    "KNN Regressor": make_pipeline(
        StandardScaler(),
        KNeighborsRegressor(n_neighbors=5)
    ),
    "XGBoost": XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        objective="reg:squarederror", random_state=42
    )
}


cv = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    cv_r2 = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
    cv_rmse = np.sqrt(
        -cross_val_score(model, X_train, y_train, cv=cv, scoring="neg_mean_squared_error")
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    test_r2 = r2_score(y_test, preds)
    test_rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"{name}:")
    print(f"  CV R²={cv_r2.mean():.4f} ± {cv_r2.std():.4f}, CV RMSE={cv_rmse.mean():.4f}")
    print(f"  Test R²={test_r2:.4f}, Test RMSE={test_rmse:.4f}\n")


  


# In[ ]:





# In[38]:


import numpy as np

results = {}

cv = KFold(n_splits=3, shuffle=True, random_state=42)

for name, model in models.items():
    cv_r2 = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")
    results[name] = np.mean(cv_r2)

sorted_models = sorted(results.items(), key=lambda x: x[1], reverse=True)

print("Best two models:")
for model_name, score in sorted_models[:2]:
    print(f"{model_name}: R2 score = {score:.4f}")


# In[41]:


from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

xgb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

xgb = XGBRegressor(random_state=42, objective='reg:squarederror')

xgb_grid = GridSearchCV(
    estimator=xgb,
    param_grid=xgb_param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1
)

xgb_grid.fit(X_train, y_train)

print("Best XGBoost Params:", xgb_grid.best_params_)
print("Best XGBoost R2 Score:", xgb_grid.best_score_)


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

gbr_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.05],
    'max_depth': [3, 4],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

gbr = GradientBoostingRegressor(random_state=42)

gbr_grid = GridSearchCV(
    estimator=gbr,
    param_grid=gbr_param_grid,
    cv=2,
    scoring='r2',
    n_jobs=-1
)

gbr_grid.fit(X_train, y_train)

print("Best Gradient Boosting Params:", gbr_grid.best_params_)
print("Best Gradient Boosting R2 Score:", gbr_grid.best_score_)


# In[ ]:

import joblib

# Save the best model (you can choose Gradient Boosting or XGBoost)
joblib.dump(xgb_grid.best_estimator_, "best_model.pkl")
print("Best model saved as best_model.pkl")



# In[ ]:



import joblib

# Example: Let's say you want to save the best XGBoost model
best_model = xgb_best_model   # replace with the variable name of your trained XGBoost

# Save the model to file
joblib.dump(best_model, "house_price_model.pkl")
print(" Best model saved as house_price_model.pkl")


# In[ ]:





# In[ ]:




