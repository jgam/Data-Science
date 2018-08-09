# Data-Science
As data scientists, our job is to extract signal from noise
jgam

## pj1
Basic tutorial as a Data Scientist

df = pd.read_csv("dataset.csv")
df.head(10)
df.describe() # sums up the data with count mean std min...(statistical values) with given columns
df['Property_Area'].value_counts()#counts the values of property area and see if there are missing values
df['ApplicantIncome'].hist(bins=50)#gets the graph of 50 bins staking ApplicantIncome data
df.boxplot(column='ApplicantIncome')#Boxplot determines where the average lies and some outliers

Categorical Variable Analysis (pivot table & cross-tabulation)
