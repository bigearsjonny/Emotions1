import os
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from keras.layers import BatchNormalization
from sklearn.metrics import mean_squared_error

#Data preprocessing
def preprocess(df):
	# convert the string of values in pixels column to a dict
	def convert_pixels_to_dict(df):
		df['pixels'] = df[' pixels'] # get rid of this stupid space
		df = df.drop([' pixels'], axis=1) # don't want this after we've converted the old ' pixels' to single columns
		return df
		
	df = convert_pixels_to_dict(df)
	
	return df

def feature_engineer(df):
	def convert_single_list_to_multiple_columns(df):
		df = df[' pixels'].str.split(' ',n=-1, expand=True)
		df.apply(pd.to_numeric)
		return df
	
	df = convert_single_list_to_multiple_columns(df)
	return df
	
#df_test = pd.read_csv('test.csv')
#df_train = pd.read_csv('train.csv')
df = pd.read_csv('icml_face_data.csv')
#df = pd.read_csv('skinny.csv', header=0, usecols=[0,2]) # use for initial testing for quick turnaround
df_Emotions = df.iloc[:, [0]].copy()
#new = old.iloc[: , [1, 6, 7]].copy() 

#df = preprocess(df) 
df = feature_engineer(df)


# Now standardize, or normalize the data
df_prescaled = df.copy() # right now, df is 18 rows with 1 column for emotion, and 2304 columns for pixel values
#print (list(df_prescaled.columns))
df_scaled = df.drop([0]) # don't scale column 0
df_scaled = scale(df_scaled)
#convert back to pandas dataframe
cols = df.columns.tolist()
cols.remove(0)
df_scaled = pd.DataFrame(df_scaled, columns=df_prescaled.columns)
#print(df_Emotions)
df_scaled[2304] = df_Emotions
df = df_scaled.copy()  # do we need deep=True
df.apply(pd.to_numeric)
#print (df[2304])
df.loc[df[2304] != 6, 2304] = 0
df.loc[df[2304] == 6, 2304] = 1

#Split data into training, testing, and validation sets
X = df.loc[:,df.columns != 2304] # X is input
y = df.loc[:, 2304] # y is target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # split the data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2) # split the data

#########################
# Build the model
#########################
model = Sequential()
# add the first hidden layer
model.add(Dense(256, activation='softmax', input_dim=2304)) # 2304 columns in the data, therefor, 2304 input_dim
# add the second hidden layer
model.add(Dense(32, activation='softmax')) 
# add the output layer
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#train the model
model.fit(X_train, y_train, epochs=200) # breaking on this line, because of stuff above

#Check Accuracy
scores = model.evaluate(X_train, y_train)
print ("Training Accuracy: %.2f%%\n" % (scores[1]*100))

scores = model.evaluate(X_test, y_test)
print ("Testing Accuracy: %.2f%%\n" % (scores[1]*100))

