import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df2=pd.read_csv("C:\\Users\\SUCHETA P\\Downloads\\_titanic.csv")
#print(df2.head(10))
#print(df2.shape)
df2=df2.drop(['Cabin'],axis=1)
df2=df2.drop(['Age'],axis=1)
#print(df2.head(10))
print(df2['Embarked'].value_counts())
df2['Embarked'].fillna('S',inplace=True)
print(df2.isnull().sum())

print(df2['Pclass'].value_counts())
df2['class']=pd.cut(df2['Pclass'],3,labels=['Upperclass','Middleclass','Lowerclass'])

fig, axes = plt.subplots(2, 2, figsize=(10, 5))
axes=axes.flatten()
for i, col in enumerate(['Age_', 'Fare', 'SibSp', 'Parch']):
    sns.histplot(df2[col], ax=axes[i])    
plt.tight_layout()  
plt.show()

fig,axes=plt.subplots(2,1,figsize=(7,6))
axes=axes.flatten()
sns.barplot(x='Sex',y='Age_',data=df2,hue='Survived',ax=axes[0])
sns.boxplot(x='class',y='Survived',data=df2,hue='Embarked',color='red',ax=axes[1])
plt.show()
sns.lineplot(y='Fare',x='Age_',data=df2)
plt.show()
print(round(df2['Fare'].mean(),3))
print(df2['Survived'].value_counts())

bins=[0,10,19,30,60,100]
df2['age_interval']=pd.cut(df2['Age_'],bins=bins,labels=['child','teenager','youngadult','middle-aged','oldaged'])
print(df2.head())
print(df2['age_interval'].value_counts())

surv_age = df2[df2['Survived'] == 1].groupby('age_interval').size()
print(surv_age)
most_survived = surv_age.idxmax()
print("Age group with the most survivors:", most_survived)

surv_class = df2[df2['Survived'] == 1].groupby('class').size()
print(surv_age)
most_survived = surv_class.idxmax()
print("class group with the most survivors:", most_survived)
