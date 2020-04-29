import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
data_csv=pd.read_csv('C:\\Users\\vkaush2\\Desktop\\ToyotaCorolla.csv',index_col=0,na_values=["??","????","####"])
copy_data=data_csv.copy()
copy_data.dropna(axis=0, inplace=True) #used to drop missing terms

#Scatter graph representation
#plt.scatter(copy_data['Age_08_04'],copy_data['Price'],c='red')
#plt.title('Price VS Age Plot')
#plt.xlabel('Age(months)')
#plt.ylabel('Price')

#histogram representation
#plt.hist(copy_data['KM'],color='green',edgecolor='white',bins=5)
#plt.title('Histogram of KM')
#plt.xlabel('KM')
#plt.ylabel('Frequecy')

#bar plot
#plt.bar(copy_data['Fuel_Type'],copy_data['Price'],color='green')

#scatter plot
#sns.regplot(x=copy_data['Age_08_04'], y=copy_data['Price'])
#sns.lmplot('Age_08_04','Price',copy_data,hue='Fuel_Type',legend=True,palette="Set1")

#Histogram
#sns.distplot(copy_data['Age_08_04'])
#sns.boxplot(y=copy_data['Price'])
#sns.pairplot(copy_data,hue="Fuel_Type")
plt.show()
