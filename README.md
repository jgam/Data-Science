# Data-Science
As data scientists, our job is to extract signal from noise
jgam

## pj1
Basic tutorial as a Data Scientist

df = pd.read_csv("dataset.csv")<br/>
df.head(10)<br/>
df.describe() # sums up the data with count mean std min...(statistical values) with given columns<br/>
df['Property_Area'].value_counts()#counts the values of property area and see if there are missing values<br/>
df['ApplicantIncome'].hist(bins=50)#gets the graph of 50 bins staking ApplicantIncome data<br/>
df.boxplot(column='ApplicantIncome')#Boxplot determines where the average lies and some outliers<br/>

Categorical Variable Analysis (pivot table & cross-tabulation)
temp1 = df['Credit_History'].value_counts(ascending=True)#Frequency table for credit History
temp2 = df.pivot_table(values='Loan_Status', index=['Credit_History'], aggfunc=lambda x: x.map({'Y':1, 'N':0}).mean())#probability of getting loan for each Credit History class

following code is ultimately what we want to learn!
```
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")
temp1.plot(kind='bar')

ax2 = fig.add_subplot(122)
temp2.plot(kind='bar')
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title('Probability of getting loan by credit history')
```

Let's take a look at this code in more details.(since two are almost identical to each other I will explore the first one.)
we use pyplot from matplotlib as plt and once we set the figure by plt.figure(figsize=(8,4)), we are setting figure size.
before we go into add_subplot, other lines make sense with setting xlabel, ylabel, and titles by calling set_ functions. Of course the last line just plots the figure with bar figure. I am sure there are different ways such as line graph and etc.
Now, what is add_subplot( )? 122 in the parameter simply means 1 row x 2 columns and 1 graph.
If there were multiple graphs, the position will be different with added subplots.
1 row and 2 column means there will be two bar graphs.

### Data Munging
This is important for two reasons.
1. There are missing values in some variables. We should estimate those values wisely depending on the amount of missing values and the expected importance of variables.
2. While looking at the distributions, we saw that ApplicantIncome and LoanAmount seemed to contain extreme values at either end. Though they might make intuitive sense, but should be treated appropriately.
3. non-numerical fields should be taken care of as well.

Now, how to check missing values in the dataset?
df.apply(lambda x: sum(x.isnull()), axis=0)
This counts null values of each column so that one can see how many values are null and apply the changes.
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
This replaces missing values(null values) to be mean of the dataset in a specific column
df['Self_Employed'].value_counts() to see answer distribution and fill it with higher probability
df['Self_Employed'].fillna('No', inplace=True)

Ok, now let's create a pivot table with median values for groups of unique values of self_employed and education features. Next, we define a function, which returns the values of these cells and apply it to fill the missing values of loan amount.
table = df.pivot_table(values='LoanAmount', index='Self_Employed', columns='Education', aggfunc=np.median)
this creates a table

### Predictive Model in Python
```
from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
	df[i] = le.fit_transform(df[i])#fit label encoder and return encoded labels
df.dtypes
```
what is LabelEncoder()?
-> from sklearn import preprocessing
-> le = preprocessing.LabelEncoder()
Before doing that, what is data normalization?
->  process of restructing a relational database in accoradance with a series of normal forms in order to reduce data redundancy and improve data integrity.


What is this?
-> efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction. It is merely used to build models and not used for reading the data, manipulating and summarizing it.
1. Supervised learning algorithms : linear Regression, support vector machines, decision trees to Bayesian methods
2. Cross-validation : check the accuracy of supervised models on unseen data
3. Unsupervised learning algorithms : clustering, factor analysis, principal component analysis to unsupervised neural networks
4. Various toy datasets : came in handy while scikit-learn.
5. Feature extraction : Useful for etracting features from images and text
Skicit-Learn requires all inputs to be numeric, we should convert all out categorical variables into numeric by encoding the categories.

##pj2 (sickit learn)

### machine learning: the problem setting
- supervised learning: data comes with additional attributes that we want to prepdict. The problem categories are classification, and regression.
- unsupervised learning consists of a set of input vectors X without any corresponding target values. The goal in such problems may be to discover groups of similar examples within the data, called as clustering, determining the distribution of data within the input space known as density estimation or to predict the data from a high-dimensional space down to two or three dimensions for the purpose of visualization.

-unsupervised learning-> training data consists of a set of input vectors x without any corresponding target values. The goal in such problems may be to discover groups of similar examples within the data, where it is called clustering, or to determine the distribution of data within the input space, known as density estimation, or to project the data from a high-dimensional space down to two or three dimensions for the purpose of visualization high-dimensional space down to two or three dimensions for the purpose of visualization.

### Loading an example dataset
```
from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()
```

-scikit-learn comes with a few standard datasets. Iris and digits datasets for classification and the boston house prices dataset for regression.
-digits.target gives the ground truth for the digit dataset, that is the number corresponding to each digit image that we are trying to learn.
```
digits.target ==> array([0,1,2, ..., 8, 9, 8])

```

shape of the data arrays: The data is always a 2D array shape (n_samples, n_features)

### Learning and predicting
```
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)
```
-Choosing the parameters of the model
we set the value of gamma manually and it is possible to automatically find good values for the parameters by using tools such as grid search and cross valiation
estimator instance **clf**, as a classifier.


### some basic tools

#### Data Operations in Pandas
Pandas handles data through Series, Data Frame, and Panel.<br/>
<br/>
Series is a one-dimensional labeled array capable of holding data of any type(integer, string, float, python objects, etc.) The axis labels are collectively called index. A pandas Series can be created using the following constructor:
```
pandas.Series(data, index, dtype, copy)
```

Constructive example is the following:
```
import pandas as pd
import numpy as np
data = np.array(['a','b','c','d'])
s = pd.Series(data)
print (s)
```

The output is the following:<br/>
0 a<br/>
1 b<br/>
2 c<br/>
3 d<br/>
dtype: object<br/>

A df is a two-dimensional data structure and data is lined up in a tabular fashion in rows and columns. A pandas DataFrame can be created using the following constructor:

```
pandas.DataFrame(data, index, columns, dtype, copy)
```

```
import pandas as pd
data = {'Name':['Tom', 'Jack','Steve', 'Ricky'], 'Age':[28,34,29,42]}
df = pd.DataFrame(data, index=['rank1', 'rank2','rank3','rank4'])
print(df)
```

The output is the following:<br/>
        Age Name
rank1	28	Tom<br/>
rank2	34	Jack<br/>
rank3	29	Steve<br/>
rank4	42	Ricky<br/>

#### Pandas Panel
*Panel* is a 3D container of data. The term Panel data is derived from econometrics and is partially responsible for the name pandas -- pan(el)-da(ta)-s.<br/>
Panel constructor is the following:<br/>
```
pandas.Panel(data, items, major_axis, minor_axis, dtype, copy)
```

The constructive example is here!
```
import pandas as pd
import numpy as np

data = {'Item1': pd.DataFrame(np.random.randn(4, 3)),
		'Item2': pd.DataFrame(np.random.randn(4, 2))}
p = pd.Panel(data)
print(p)
```