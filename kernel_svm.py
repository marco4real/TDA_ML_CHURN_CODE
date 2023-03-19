# Kernel SVM

# Importing the libraries

import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def Kernel_SVM():

    # Importing the dataset
    dataset = pd.read_csv('Dataset_Statistics_15_percent_random.txt', header = None)
    dataset.columns = 'Column' + dataset.columns.astype(str)
 
    X = dataset.iloc[:, :24].values
    y = dataset.iloc[:, 24].values

    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Fitting Kernel SVM to the Training set
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf', random_state = 0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    accuracy = cm.trace()/cm.sum()
    print("Accuracy: %.2f%%" % (accuracy*100.00))
    
    # Compute the precision
    precision = precision_score(y_test, y_pred, average='weighted')
    print("Precision: %.2f%%" % (precision*100.00))
    
    # Compute the recall
    recall = recall_score(y_test, y_pred, average='weighted')
    print("Recall: %.2f%%" % (recall*100.00))
    
    # Compute the precision
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("F-Measure: %.2f%%" % (f1*100.00))
    
    return accuracy, precision, recall, f1

if __name__ == '__main__':
    Kernel_SVM()

