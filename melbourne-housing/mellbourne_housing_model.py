import pandas as pd 
from sklearn.tree import DecisionTreeRegressor 

#housing data file name csv
mel_file_path = "melbourne-housing\melb_data.csv"

#storing the housing data 
mel_data = None

mel_data = pd.read_csv(mel_file_path)

#print(mel_data.describe())

#removing zeros in the 
mel_data = mel_data.dropna(axis=0)

#Housing data columbns 
#y = mel_data.columns

#the housing features 
y = mel_data.Price


feature_names = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']

X = mel_data[feature_names]

#definition of the model random state =1 
mel_model = DecisionTreeRegressor(random_state=1)

#fit model
mel_model.fit(X,y)

#making predictions of the 5 houses 
print("Making predictions for the following 5 houses ")
print(X.head())

#the prediction for the five houses 
print("The predictions ")

print(mel_model.predict(X.head()))
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
