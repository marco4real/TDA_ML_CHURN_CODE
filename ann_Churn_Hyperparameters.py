# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

def Neural_Network():
    # Importing the dataset
    dataset = pd.read_csv('Churn_tuned_dataset.csv', header = None)
    dataset.columns = 'Column' + dataset.columns.astype(str)
    

    X = dataset.iloc[:, :31].values
    y = dataset.iloc[:, 31].values
    
    #Nonormalising the output data
    min_max_scaler = preprocessing.MinMaxScaler()
    y = min_max_scaler.fit_transform(y.reshape(-1,1))
    #norm_dataset = pd.DataFrame(dataset_scaled)

    #from keras.utils import to_categorical
    #y = to_categorical(y)

    # Encoding categorical data
    #from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    #labelencoder_X_1 = LabelEncoder()
    #X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
    #labelencoder_X_2 = LabelEncoder()
    #X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
    #onehotencoder = OneHotEncoder(categorical_features = [1])
    #X = onehotencoder.fit_transform(X).toarray()
    #X = X[:, 1:]

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Part 2 - Now let's make the ANN!

    # Importing the Keras libraries and packages
    import keras
    from keras.models import Sequential
    from keras.layers import Dense

    # Initialising the ANN
    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units = 31, kernel_initializer = 'uniform', activation = 'relu', input_dim = 31))

    # Adding the second hidden layer
    classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))

    # Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'softmax'))

    # Compiling the ANN
    opt = keras.optimizers.Adam(learning_rate=0.01)
    classifier.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])

    # Fitting the ANN to the Training set
    classifier.fit(X_train, y_train, batch_size = 50, epochs = 10)

    # Part 3 - Making predictions and evaluating the model

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred >= 0.5)

     # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    # Compute the accuracy
    accuracy = cm.trace()/cm.sum()
   
    print("====================================================================")

    return accuracy

if __name__ == '__main__':
    Neural_Network()


