# ===================Question 1======================
import pandas as pd

start_ups = pd.read_csv(r"D:\360DigiTMG\DataScience\32. ANN-MLP\Assignment\50_Startups (2).csv");
y = start_ups.Profit
X = start_ups.iloc[:,:4]

X_numeric = X.iloc[:, :3]
X_string = X.select_dtypes('object')

from sklearn.preprocessing import MinMaxScaler
mnx = MinMaxScaler()
X_numeric = pd.DataFrame(mnx.fit_transform(X_numeric))

from sklearn.preprocessing import LabelEncoder

lbl_enc = LabelEncoder();
X_string.State = pd.DataFrame(lbl_enc.fit_transform(X_string.State))

X_normalised = pd.concat([X_numeric, X_string], axis=1)

X_normalised = X_normalised.iloc[:,:].values.astype("float32")
y = y.values.astype("float32")

# divide the data into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_normalised, y, test_size=0.2)


from keras.models import Sequential
from keras.layers import Dense

num_of_classes = y_test.shape
num_of_classes
model = Sequential()
model.add(Dense(10,input_dim =X_train.shape[1],activation="relu"))
model.add(Dense(11,activation="relu"))
model.add(Dense(5,activation="relu"))
model.add(Dense(4,activation="relu"))
model.add(Dense(1,activation="linear"))
model.compile(loss="mse",optimizer="adam", metrics=['mean_absolute_error'])

model.fit(x=X_train,y=y_train,batch_size=32,epochs=100)

eval_score_test = model.evaluate(X_test,y_test,verbose = 1)


# ===================Question 2======================
import pandas as pd

fireforests = pd.read_csv(r"D:\360DigiTMG\DataScience\32. ANN-MLP\Assignment\fireforests.csv");
y = fireforests.area
X = fireforests.drop(['area'], axis=1)

X_numeric = X.select_dtypes(['int64', 'float64'])
X_string = X.select_dtypes('object')

from sklearn.preprocessing import MinMaxScaler
mnx = MinMaxScaler()
X_numeric = pd.DataFrame(mnx.fit_transform(X_numeric))

X_string = pd.get_dummies(X_string, drop_first=True)

X_normalised = pd.concat([X_numeric, X_string], axis=1)

X_normalised = X_normalised.iloc[:,:].values.astype("float32")
y = y.values.astype("float32")

# divide the data into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_normalised, y, test_size=0.2)


from keras.models import Sequential
from keras.layers import Dense
import tensorflow.keras.backend as K

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


num_of_classes = y_test.shape
num_of_classes
model = Sequential()
model.add(Dense(10,input_dim =X_train.shape[1],activation="relu"))
model.add(Dense(11,activation="relu"))
model.add(Dense(5,activation="relu"))
model.add(Dense(4,activation="relu"))
model.add(Dense(1,activation="linear"))
model.compile(loss="mse",optimizer="adam", metrics=[rmse])

model.fit(x=X_train,y=y_train,batch_size=32,epochs=100)

eval_score_test = model.evaluate(X_test,y_test,verbose = 1)


# ===================Question 3======================
import pandas as pd

concrete = pd.read_csv(r"D:\360DigiTMG\DataScience\32. ANN-MLP\Assignment\concrete.csv");
y = concrete.strength
X = concrete.drop(['strength'], axis=1)

from sklearn.preprocessing import MinMaxScaler
mnx = MinMaxScaler()
X = pd.DataFrame(mnx.fit_transform(X))

X = X.iloc[:,:].values.astype("float32")
y = y.values.astype("float32")

# divide the data into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


from keras.models import Sequential
from keras.layers import Dense
import tensorflow.keras.backend as K

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


num_of_classes = y_test.shape
num_of_classes
model = Sequential()
model.add(Dense(10,input_dim =X_train.shape[1],activation="relu"))
model.add(Dense(11,activation="relu"))
model.add(Dense(5,activation="relu"))
model.add(Dense(4,activation="relu"))
model.add(Dense(1,activation="linear"))
model.compile(loss="mse",optimizer="adam", metrics=[rmse])

model.fit(x=X_train,y=y_train,batch_size=32,epochs=100)

eval_score_test = model.evaluate(X_test,y_test,verbose = 1)


# ===================Question 4======================
import pandas as pd

rpl = pd.read_csv(r"D:\360DigiTMG\DataScience\32. ANN-MLP\Assignment\RPL.csv");
y = rpl.Exited
X = rpl.drop(['Exited'], axis=1)

X_numeric = X.select_dtypes(['int64', 'float64'])
X_string = X.select_dtypes('object')

from sklearn.preprocessing import MinMaxScaler
mnx = MinMaxScaler()
X_numeric = pd.DataFrame(mnx.fit_transform(X_numeric))

X_string = pd.get_dummies(X_string, drop_first=True)

X_normalised = pd.concat([X_numeric, X_string], axis=1)

X_normalised = X_normalised.iloc[:,:].values.astype("float32")
y = y.values.astype("float32")


# divide the data into train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_normalised, y, test_size=0.2)

from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

from keras.models import Sequential
from keras.layers import Dense


num_of_classes = y_test.shape[1]
num_of_classes
model = Sequential()
model.add(Dense(210,input_dim =X_train.shape[1],activation="relu"))
model.add(Dense(111,activation="tanh"))
model.add(Dense(150,activation="tanh"))
model.add(Dense(100,activation="tanh"))
model.add(Dense(num_of_classes,activation="softmax"))
model.compile(loss="categorical_crossentropy",optimizer="adam", metrics=['accuracy'])

model.fit(x=X_train,y=y_train,batch_size=32,epochs=100)

eval_score_test = model.evaluate(X_test,y_test,verbose = 1)
