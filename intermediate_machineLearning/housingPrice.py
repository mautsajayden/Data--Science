import os
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(__file__)

#read the data 
X_full = pd.read_csv(os.path.join(BASE_DIR, "train.csv"),index_col='Id')
X_test_full = pd.read_csv(os.path.join(BASE_DIR, "test.csv"),index_col='Id')

#obtaining the features and the target predictiosn 

y = X_full.SalePrice

features =  ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = X_full[features].copy()
X_test = X_test_full[features].copy()

#break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=0)

print(X_train.head())

from sklearn.ensemble import RandomForestRegressor

#define the models 

model1 = RandomForestRegressor(n_estimators=50, random_state=0)
model2 = RandomForestRegressor(n_estimators=100, random_state= 0)
model3 = RandomForestRegressor(n_estimators= 100, criterion='absolute_error',random_state=0)
model4 = RandomForestRegressor(n_estimators=200, min_samples_split=20,random_state= 0 )
model5 = RandomForestRegressor(n_estimators= 100, max_depth=7, random_state=0)

#list with models 
models = [model1,model2,model3,model4,model5]

from sklearn.metrics  import mean_absolute_error

def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t,y_t)
    predictions = model.predict(X_v)
    return mean_absolute_error(y_v,predictions)    

score_mae = []

for i in range(0, len(models)):
    mae = score_model(models[i])
    score_mae.append(mae)
    print(f"Model: {i+1} MAE: {mae}")
    
#the best model
min_mae = min(score_mae)
best_model = [ x  for x in range(len(score_mae)) if score_mae[x] == min_mae ]
 
print(f'The best model: {best_model[0]+1} , MAE = {min_mae} ')