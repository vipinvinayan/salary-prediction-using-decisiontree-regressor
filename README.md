# salary-prediction-using-decisiontree-regressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score,KFold
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error


data=pd.read_csv(r"C:\Users\Admin\Desktop\Salary_Data.csv")
print(data)

data.info()

duplicate_rows = data.duplicated()

print(data[data.duplicated()].shape)
data1=data.drop_duplicates()

data1

print(data1.isnull().sum())

data2=data1.dropna()

print(data2)

print(data2.head())

print(data1['Gender'].value_counts())
print(data1['Gender'].value_counts().plot(kind='bar'))
plt.title('Distribution of Gender ')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

data2 = data2[data2['Gender']!='Other']
print(data2['Gender'].value_counts())

print(data2['Gender'].value_counts())
print(data2['Gender'].value_counts().plot(kind='bar'))
plt.title('Distribution of Gender ')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

print(data2['Education Level'].value_counts())
print(data2['Education Level'].value_counts().plot(kind='bar'))
plt.title('Distribution of Education Levels')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.show()

replace_dict = {'phD': 'PhD',"Bachelor's Degree": "Bachelor's","Master's Degree" : "Master's"}
data2['Education Level'] = data2['Education Level'].replace(replace_dict)

print(data2['Education Level'].value_counts())

print(data2['Education Level'].value_counts().plot(kind='bar'))
plt.title('Distribution of Education Levels')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.show()

print(data2)

print(data2["Job Title"].unique())

print(data2['Job Title'].value_counts())
print(data2['Job Title'].value_counts().plot(kind='bar'))
plt.title('Distribution of Job Title')
plt.xlabel('Job Title')
plt.ylabel('Count')
plt.show()

print(data2['Job Title'].value_counts()[:11])


print(data2['Job Title'].value_counts()[:11].plot(kind='bar'))
plt.title('Distribution of Job Title')
plt.xlabel('Job Title')
plt.ylabel('Count')
plt.show()

data3= ['Software Engineer Manager', 'Full Stack Engineer', 'Senior Project Engineer', 'Senior Software Engineer', 'Data Scientist', 'Back end Developer', 'Software Engineer', 'Front end Developer', 'Marketing Manager', 'Product Manager', 'Data Analyst']
data4 = data2[data2['Job Title'].isin(data3)]
print(data4.info())

plt.scatter(data4["Years of Experience"] , data4["Salary"])
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("years of experience vs Salary")
plt.show()

plt.scatter(data4["Age"] , data4["Salary"])
plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("Age vs Salary")
plt.show()


label_encoder = LabelEncoder()
data4['Gender_Encode'] = label_encoder.fit_transform(data4['Gender'])
print(data4[['Gender', 'Gender_Encode']])
data4['qualification_Encode'] = label_encoder.fit_transform(data4['Education Level'])
print(data4[['Education Level', 'qualification_Encode']])
data4['JobTitle_Encode'] = label_encoder.fit_transform(data4['Job Title'])
print(data4[['Job Title', 'JobTitle_Encode']])
print(data4.head())


d_col=['Gender','Education Level','Job Title']
data5=data4.drop(columns=d_col)
print(data5.info())

data5.head()

scaler=StandardScaler()
data5_scaled=scaler.fit_transform(data5)
plt.boxplot(data5_scaled)
plt.title('Boxplot of Scaled Data')
plt.show()

corr=data5[['Age','Years of Experience','Gender_Encode','qualification_Encode','JobTitle_Encode','Salary']].corr()
print(corr)
sns.heatmap(corr,annot=True)
plt.show()

x = data5[['Age', 'Years of Experience']]
y = data5['Salary']
print(x)
print(y)

models=[]
models.append(('KNN',KNeighborsRegressor()))
models.append(("LNN",LinearRegression()))
models.append(('DT',DecisionTreeRegressor()))
models.append(('LS',Lasso()))
models.append(('RD',Ridge()))
result1=[]
names=[]
scoring = 'neg_mean_squared_error'
Kfold= KFold(n_splits=10, shuffle=True, random_state=42)
for name, model in models:
    cv_results = cross_val_score(model, x, y, cv=Kfold, scoring=scoring)
    result1.append(cv_results)
    names.append(name)
    print(f"MSE of {name}: {cv_results.mean()}")
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(result1)
ax.set_xticklabels(names)
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

Tree_Model= DecisionTreeRegressor(max_depth=2)
print(Tree_Model.fit(x_train, y_train))
result2=Tree_Model.predict(x_test)
print(y)
print(result2)

print(mean_squared_error(y_test,result2))
print(np.sqrt(mean_squared_error(y_test,result2)))
print(mean_absolute_error(y_test,result2))
print(r2_score(y_test,result2))


plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
plt.scatter(x=y_test,y=result2,color='black')
plt.title('predicted point vs actual point',fontdict={'fontsize':15})
plt.xlabel('Actual points (y_test)',fontdict={'fontsize':10})
plt.ylabel('Predicted points (result2)',fontdict={'fontsize':10})
plt.show()


plt.figure(figsize=(15, 10))
plot_tree(Tree_Model, feature_names=x.columns.tolist(), filled=True, rounded=True)
plt.show()

new_data={'Age': 30,'Years of Experience':8}

new_data_df=pd.DataFrame([new_data])

scaler = MinMaxScaler()

x_scaled=scaler.fit_transform(new_data_df)

predicted_salary=Tree_Model.predict(x_scaled)
print("predicted salary for employee:",predicted_salary)

