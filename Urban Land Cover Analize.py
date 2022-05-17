"""
Mehmet Nejat Baturay 18120205040
Egemen Yapucu 18120205027
Tevfik Gürhan Kuraş 19120205062
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris 
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier 
from sklearn.naive_bayes import GaussianNB 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
import  sklearn.metrics as metrics
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
url = 'https://drive.google.com/file/d/17xyJID13Uq4UcivZT4E54L1rwmkodbvu/view?usp=sharing'
url1 = 'https://drive.google.com/uc?id=' + url.split('/')[-2]
url = 'https://drive.google.com/file/d/1tQuOqewUX_2zsn7OlB4GC6kXArSfQ9S6/view?usp=sharing'
url2 = 'https://drive.google.com/uc?id=' + url.split('/')[-2]

testing = pd.read_csv(url1)
training = pd.read_csv(url2)

data=pd.concat([testing,training], ignore_index=True)

X = pd.DataFrame()

y = pd.DataFrame()


X=data

X=X.drop(columns=["class"])


y["class"]=data["class"]

#z-score normalizasyonu  uygulanmadan verinin işlenmesi
#-------------------------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1, test_size=0.3)

clfMLP2 = MLPClassifier(hidden_layer_sizes  = (20,70,120),
                    activation          = "identity",
                    random_state        = 1,
                    max_iter            = 950,
                    verbose             = 1)





clfMLP = MLPClassifier(hidden_layer_sizes  = (100,150,200),
                    activation          = "tanh",
                    random_state        = 1,
                    max_iter            = 950,
                    verbose             = 1)



scoringlist = ["accuracy","precision_macro","recall_macro","f1_macro"]



dict2_5 = cross_validate(clfMLP2,X_train,y_train.values.ravel(),scoring=scoringlist,cv=5)


dict2_10 = cross_validate(clfMLP2,X_train,y_train.values.ravel(),scoring=scoringlist,cv=10)

dict1_5 = cross_validate(clfMLP,X_train,y_train.values.ravel(),scoring=scoringlist,cv=5)

dict1_10 = cross_validate(clfMLP,X_train,y_train.values.ravel(),scoring=scoringlist,cv=10)


print("\n")
print(np.array(dict1_5["test_accuracy"]).mean())
print(np.array(dict1_5["test_precision_macro"]).mean())
print(np.array(dict1_5["test_recall_macro"]).mean())
print(np.array(dict1_5["test_f1_macro"]).mean())

print("\n")
print(np.array(dict1_10["test_accuracy"]).mean())
print(np.array(dict1_10["test_precision_macro"]).mean())
print(np.array(dict1_10["test_recall_macro"]).mean())
print(np.array(dict1_10["test_f1_macro"]).mean())

print("\n")
print(np.array(dict2_5["test_accuracy"]).mean())
print(np.array(dict2_5["test_precision_macro"]).mean())
print(np.array(dict2_5["test_recall_macro"]).mean())
print(np.array(dict2_5["test_f1_macro"]).mean())

print("\n")
print(np.array(dict2_10["test_accuracy"]).mean())
print(np.array(dict2_10["test_precision_macro"]).mean())
print(np.array(dict2_10["test_recall_macro"]).mean())
print(np.array(dict2_10["test_f1_macro"]).mean())






scoringlist = ["accuracy","precision_macro","recall_macro","f1_macro"]




gnb = GaussianNB()

clfGNB = gnb.fit(X_train,y_train.values.ravel())


print(cross_validate(clfGNB,X_train,y_train.values.ravel(),scoring=scoringlist,cv=5))
print(cross_validate(clfGNB,X_train,y_train.values.ravel(),scoring=scoringlist,cv=10))


#z-score normalizasyonu uygulandıktan sonra verinin işlenmesi
#--------------------------------------------------------------------------------

standartscaler = StandardScaler()

X=data

X=X.drop(columns=["class"])

labelencoder = LabelEncoder()

data["class"]= labelencoder.fit_transform(data["class"].values)


X = standartscaler.fit_transform(data)


y["class"]=data["class"]




X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1, test_size=0.3)

clfMLP2 = MLPClassifier(hidden_layer_sizes  = (20,70,120),
                    activation          = "identity",
                    random_state        = 1,
                    max_iter            = 950,
                    verbose             = 1)





clfMLP = MLPClassifier(hidden_layer_sizes  = (100,150,200),
                    activation          = "tanh",
                    random_state        = 1,
                    max_iter            = 950,
                    verbose             = 1)



scoringlist = ["accuracy","precision_macro","recall_macro","f1_macro"]



dict1_5 = cross_validate(clfMLP,X_train,y_train.values.ravel(),scoring=scoringlist,cv=5)

dict1_10 = cross_validate(clfMLP,X_train,y_train.values.ravel(),scoring=scoringlist,cv=10)

dict2_5 = cross_validate(clfMLP2,X_train,y_train.values.ravel(),scoring=scoringlist,cv=5)

dict2_10 = cross_validate(clfMLP2,X_train,y_train.values.ravel(),scoring=scoringlist,cv=10)

print("\n")
print(np.array(dict1_5["test_accuracy"]).mean())
print(np.array(dict1_5["test_precision_macro"]).mean())
print(np.array(dict1_5["test_recall_macro"]).mean())
print(np.array(dict1_5["test_f1_macro"]).mean())

print("\n")
print(np.array(dict1_10["test_accuracy"]).mean())
print(np.array(dict1_10["test_precision_macro"]).mean())
print(np.array(dict1_10["test_recall_macro"]).mean())
print(np.array(dict1_10["test_f1_macro"]).mean())

print("\n")
print(np.array(dict2_5["test_accuracy"]).mean())
print(np.array(dict2_5["test_precision_macro"]).mean())
print(np.array(dict2_5["test_recall_macro"]).mean())
print(np.array(dict2_5["test_f1_macro"]).mean())

print("\n")
print(np.array(dict2_10["test_accuracy"]).mean())
print(np.array(dict2_10["test_precision_macro"]).mean())
print(np.array(dict2_10["test_recall_macro"]).mean())
print(np.array(dict2_10["test_f1_macro"]).mean())




gnb = GaussianNB()

clfGNB = gnb.fit(X_train,y_train.values.ravel())



print(cross_validate(clfGNB,X_train,y_train.values.ravel(),scoring=scoringlist,cv=5))

print(cross_validate(clfGNB,X_train,y_train.values.ravel(),scoring=scoringlist,cv=10))
