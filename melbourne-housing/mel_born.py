import pandas as p

mel_cvs = pd.read_csv("melbourne-housing\melb_data.csv\melb_data.csv")

#print(mel_cvs.describe()

#shows the columns of the data set 
mel_cvs.columns

#drops all the missing values 
mel_cvs = mel_cvs.dropna(axis=0)

#prediction target 
y = mel_cvs.Price

#melbourne features 
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

x = mel_cvs[melbourne_features]

