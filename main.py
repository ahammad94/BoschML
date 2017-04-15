from sklearn import metrics
from sklearn.metrics import precision_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
Temp=pd.DataFrame()
Temp_Final=pd.DataFrame()
for i in range (0,50):
    train_numeric = pd.read_csv('train_numeric.csv', skiprows=range(1,i*20000), nrows=20000, low_memory=False)
    Temp=train_numeric.loc[train_numeric['Response'] ==1]
    frames=[Temp_Final,Temp]
    Temp_Final=pd.concat(frames)
K=len(Temp_Final.index)
print(K)
Temp_Final0=pd.DataFrame()
test_numeric = pd.read_csv('train_numeric.csv', skiprows=range(1,15000), nrows=K, low_memory=False)
#print(test_numeric)
Test_Len=len(test_numeric.index)
print(Test_Len)
frames=[test_numeric,Temp_Final]
Train_Final=pd.concat(frames)
Train_Final
from sklearn.model_selection import train_test_split
R=Train_Final.columns
for j in range(0,969):
        mean=Train_Final[R[j]].mean()
        Train_Final[R[j]]=Train_Final[R[j]].fillna(mean)
        mean=test_numeric[R[j]].mean()
        test_numeric[R[j]]=test_numeric[R[j]].fillna(mean)
Train_Response=Train_Final['Response']
del Train_Final['Response']
Test_Response=test_numeric['Response']
del test_numeric['Response']
X=Train_Final
y=Train_Response
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
from sklearn.ensemble import RandomForestClassifier
clf1=RandomForestClassifier(n_estimators=100)
clf1.fit(X_train,y_train)
Ans=pd.DataFrame(clf1.predict(X_test))
print(Ans)
clf1.score(X_test,y_test)
#clf1.score(test_numeric,Test_Response)
fpr, tpr, thresholds = metrics.roc_curve(y_test, Ans)
print(fpr)
print(tpr)
print(thresholds)
FPR1=plt.plot(fpr, label="FPR")  #Blue
TPR1=plt.plot(tpr, label="TPR")  #Green
#plt.legend(handles=[FPR1, TPR1],loc='best')
plt.show()
precision_score(y_test, Ans, average='macro')
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.show()