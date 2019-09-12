#data preprocessing

#classification into whether or not a person buys from into age,sal

#imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#getting dataset
dataset=pd.read_csv('Churn_Modelling.csv')
X=dataset.iloc[:,3:13].values
Y=dataset.iloc[:,13].values

#mising data - fine
#categorical to numerical
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X1=LabelEncoder()
X[:,1]=labelencoder_X1.fit_transform(X[:,1])
labelencoder_X2=LabelEncoder()
X[:,2]=labelencoder_X2.fit_transform(X[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]
labelencoder_Y=LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)

#splitting
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

#scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#Training and Evaluating the ANN using K-Fold Cross Validation

#Step -1 Imports
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential 
from keras.layers import Dense
def build_classifier(optimizer):
    classifier=Sequential()
    classifier.add(Dense(output_dim=6,input_dim=11,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
    classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
classifier=KerasClassifier(build_fn=build_classifier)
parameters={'batch_size':[25,32],
             'epochs':[100,500],
             'optimizer':['adam','rmsprop']}
grid_search=GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv=10)
grid_search=grid_search.fit(X=X_train,y=Y_train)
best_parameters=grid_search.best_params_
best_accuracy=grid_search.best_score_
