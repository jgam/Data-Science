# Data-Science
As data scientists, our job is to extract signal from noise
jgam

## pj1
Basic tutorial as a Data Scientist

⋅⋅*df = pd.read_csv("dataset.csv")
⋅⋅*df.head(10)
⋅⋅*df.describe() # sums up the data with count mean std min...(statistical values) with given columns
⋅⋅*df['Property_Area'].value_counts()#counts the values of property area and see if there are missing values
⋅⋅*df['ApplicantIncome'].hist(bins=50)#gets the graph of 50 bins staking ApplicantIncome data
⋅⋅*df.boxplot(column='ApplicantIncome')#Boxplot determines where the average lies and some outliers

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



