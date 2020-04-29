#kaggle project TITANIC
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
#from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
test=pd.read_csv("C:\\Users\\vkaush2\\Desktop\\titanic\\test.csv")
train=pd.read_csv("C:\\Users\\vkaush2\\Desktop\\titanic\\train.csv")
test.info()
train.info()
test.describe()  
missing=test.isnull().mean() #or isnull.sum()
#sns.barplot(y='Survived',x='Sex',data=train)
#Categorical data is considered as feature
#Pclass,Sex,SibSp,Parch,Embarked,Cabin
def bar_chart(feature):
    survived=train[train['Survived']==1][feature].value_counts()
    dead=train[train['Survived']==0][feature].value_counts()
    df=pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar')
""" train.groupby(['Survived'])[feature].value_counts().plot.bar() #another method to group variables"""
bar_chart('Sex')
bar_chart('Pclass')
bar_chart('SibSp')
bar_chart('Parch')
#similarly we can visualize the relation b/w survivved and other categorical features.
train_test_data=[train,test]
#print(train_test_data)
for dataset in train_test_data:
    dataset['Title']=dataset['Name'].str.extract(' ([A-Za-z]+)\. ',expand=False)
#    test['Title'].value_counts()
#   train['Title'].value_counts()
title_mapping={"Mr":0,"Miss":1,"Mrs":2,"Master":3,
               "Dr":3,"Rev":3,"Col":3,"Mlle":3,"Ms":3,"Major":3,"Sir":3,"Lady":3,"Capt":3,"Mme":3,"Don":3,"Countess":3,"Jonkheer":3}
for dataset in train_test_data:
    dataset['Title']=dataset['Title'].map(title_mapping)
bar_chart('Title')
train.drop('Name',axis=1,inplace=True)
test.drop('Name',axis=1,inplace=True)
sex_mapping={'male':0,'female':1}
for dataset in train_test_data:
    dataset['Sex']=dataset['Sex'].map(sex_mapping)
bar_chart('Sex')
#fill missing age with median age for eaach title (Mr,Mrs,Miss,others)
train['Age'].fillna(train.groupby('Title')['Age'].transform('median'),inplace=True)
test['Age'].fillna(train.groupby('Title')['Age'].transform('median'),inplace=True)
#Facet graph to show age and survival relation
facet=sns.FacetGrid(train,hue='Survived',aspect=4)
facet.map(sns.kdeplot,'Age',shade=True)
facet.set(xlim=(0,train['Age'].max()))
facet.add_legend()
plt.show()
#for particular X-Axis range use plt.xlim(0,20)

#We use binning technique to set a range for Age values it will help to find the survival pattern more
for dataset in train_test_data:
    dataset.loc[dataset['Age'] <=16, 'Age'] = 0,
    dataset.loc[(dataset['Age'] >16) & (dataset['Age']<=26),'Age'] = 1,
    dataset.loc[(dataset['Age'] >26) & (dataset['Age']<=36),'Age'] = 2,
    dataset.loc[(dataset['Age'] >36) & (dataset['Age']<=46),'Age'] = 3,
    dataset.loc[(dataset['Age'] >46) & (dataset['Age']<=56),'Age'] = 4,
    dataset.loc[dataset['Age'] >56, 'Age']=5
bar_chart('Age')
bar_chart('Embarked')
#In this code we change embark values and fill the missing terms with frequently repeating term.
embark={'S':0,'C':1,'Q':2}
for dataset in train_test_data:
    dataset['Embarked']=dataset['Embarked'].map(embark)
for dataset in train_test_data:
    dataset['Embarked']=dataset['Embarked'].fillna('0')
#Code to combine 2 columns together to find something usefull.
Pclass1 = train[train['Pclass']==1]['Embarked'].value_counts()
Pclass2 = train[train['Pclass']==2]['Embarked'].value_counts()
Pclass3 = train[train['Pclass']==3]['Embarked'].value_counts()
df=pd.DataFrame([Pclass1,Pclass2,Pclass3])
df.index=['1st class','2nd class','3rd class']
df.plot(kind='bar',stacked=True)

#filling missing fare with median fare with respect to Pclass.filling missing value in fare with respect to Pclass will give us more accurate data because differnt class has differnt fare range. Getting a median respective to class will give more accurate result
train['Fare'].fillna(train.groupby('Pclass')['Fare'].transform('median'),inplace=True)    
test['Fare'].fillna(test.groupby('Pclass')['Fare'].transform('median'),inplace=True)
#facet graph for above situation
facet = sns.FacetGrid(train,hue='Survived',aspect=4)
facet.map(sns.kdeplot,'Fare',shade=True)
facet.set(xlim=(0,train['Fare'].max()))
facet.add_legend()
plt.show()
#from graph we can see people travelling with cheap tickets died more. 
#Binning in Fare
for dataset in train_test_data:
    dataset.loc[(dataset['Fare']<=20), 'Fare']=0,
    dataset.loc[(dataset['Fare']>20) & (dataset['Fare']<=40),'Fare']=1
    dataset.loc[(dataset['Fare']>40) & (dataset['Fare']<=100),'Fare']=2
    dataset.loc[(dataset['Fare']>100),'Fare']=3
train.Cabin.value_counts()
#cabin column has numeric+alhabet number we are extracting alphabet.
for dataset in train_test_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]
    print(dataset)
Pclass1 = train[train['Pclass']==1]['Cabin'].value_counts()
Pclass2 = train[train['Pclass']==2]['Cabin'].value_counts()
Pclass3 = train[train['Pclass']==3]['Cabin'].value_counts()
df=pd.DataFrame([Pclass1,Pclass2,Pclass3])
df.index=['1st class','2nd class','3rd class']
df.plot(kind='bar',stacked=True)  
cabin_mapping={'A':0,'B':0.4,'C':0.8,'D':1.2,'E':1.6,'F':2,'G':2.4,'T':2.8} 
for dataset in train_test_data:
    dataset['Cabin']=dataset['Cabin'].map(cabin_mapping)
#fill missing fare with median fare for each Pclass
train['Cabin'].fillna(train.groupby('Pclass')['Cabin'].transform('median'),inplace=True)    
test['Cabin'].fillna(test.groupby('Pclass')['Cabin'].transform('median'),inplace=True)
#Getting info from family size
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
facet=sns.FacetGrid(train,hue='Survived',aspect=4)
facet.map(sns.kdeplot,'FamilySize',shade=True)
facet.set(xlim=(0,train['FamilySize'].max()))
facet.add_legend()
plt.show()
family_mapping={1:0,2:0.4,3:0.8,4:1.2,5:1.6,6:2,7:2.4,8:2.8,9:3.2,10:3.6,11:4} 
for dataset in train_test_data:
    dataset['FamilySize']=dataset['FamilySize'].map(family_mapping)
features_drop=['Ticket','SibSp','Parch']
train=train.drop(features_drop,axis=1)
test=test.drop(features_drop,axis=1)
train=train.drop(['PassengerId'],axis=1)
train_data=train.drop('Survived',axis=1)
title_category=test_data['Title'].value_counts()
test_data['Title'].fillna(title_category.index[0],inplace=True)
title_category2=test['Title'].value_counts()
test['Title'].fillna(title_category2.index[0],inplace=True)

target=train['Survived']

##Modelling
#Cross Validation(K-Fold)
k_fold = KFold(n_splits=10,shuffle=True, random_state=0)
#SVC model
clf= SVC()
scoring='accuracy'
score= cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)
round(np.mean(score)*100,2)

#Testing
clf=SVC()
clf.fit(train_data,target)
test_data=test.drop('PassengerId',axis=1).copy()
prediction= clf.predict(test_data)
submission=pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':prediction})
submission.to_csv('submission.csv',index=False)
submission = pd.read_csv('submission.csv')
submission.head()

"""Checking some other techniques of feature engineering"""
#Feature engineering of Cabin
train['Cabin_reduced']=train['Cabin'].astype(str).str[0]
len(train['Cabin_reduced'].unique())
test['Cabin_reduced']=test['Cabin'].astype(str).str[0]
len(test['Cabin_reduced'].unique())
train.groupby('Cabin_reduced')['Pclass'].median().plot.bar()
plt.xlabel(cabin)
plt.ylabel('Pclass')
plt.title(cabin)
plt.show()
train.groupby('Pclass')['Fare'].median().plot.bar()
train.groupby('Embarked')['Pclass'].median().plot.bar() #No relation
train.groupby('Cabin_reduced')['Survived'].median().plot.bar()
t

#feature engineering of age
for dataset in train:
    train['Title']=train['Name'].str.extract(' ([A-Za-z]+)\. ',expand=False)
len(train['Title'].unique())
train['Title'].head(10).unique()
train['Age'].fillna(train.groupby('Title')['Age'].transform('median'),inplace=True)
test['Age'].fillna(train.groupby('Title')['Age'].transform('median'),inplace=True)
    
