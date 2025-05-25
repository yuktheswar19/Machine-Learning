# **Basic Preprocessing Template**
collapsed:: true
	- ```python
	  import numpy as np
	  import matplotlib.pyplot as plt
	  import pandas as pd
	  
	  # Importing the dataset
	  
	  dataset = pd.read_csv('Data.csv')
	  X = dataset.iloc[:, :-1].values
	  y = dataset.iloc[:, -1].values
	  
	  # Splitting the dataset into the Training set and Test set
	  
	  from sklearn.model_selection import train_test_split
	  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
	  ```
	-
		- **Reads a CSV file and separates features (X) and target (y).**
			- ```python
			  dataset = pd.read_csv('Data.csv')
			  X = dataset.iloc[:, :-1].values
			  y = dataset.iloc[:, -1].values
			  ```
		- **Splits data into 80% training and 20% test sets.**
		- ```python
		  from sklearn.model_selection import train_test_split
		  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
		  ```
-
- # **Handling Missing Values**
  collapsed:: true
	- ```python
	  from sklearn.impute import SimpleImputer
	  imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
	  
	  # Selection all Rows from x [ " : "]
	  # Selection columns from 1 up to , excluding 3 [" :3 "]
	  imputer.fit(X[:, 1:3])
	  
	  X[:, 1:3] = imputer.transform(X[:, 1:3])
	  print(X)
	  ```
		-
		- **Import**: Imports SimpleImputer from Scikit-learn to replace missing values.
		- **Initialize Imputer**: Creates an imputer that replaces np.nan (missing values) with the mean of the column, focusing on columns 1 and 2 of X (index 1:3).
		- **Fit**: Learns the mean values for columns 1 and 2 using fit.
		- **Transform**: Replaces missing values in those columns with the learned means and updates X.
		- **Print**: Displays the updated X with no missing values in columns 1 and 2.
-
- # **Encoding Categorical Variables**
  collapsed:: true
	- ```python
	  from sklearn.compose import ColumnTransformer
	  from sklearn.preprocessing import OneHotEncoder
	  
	  ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], 
	                         remainder='passthrough')
	  
	  X = np.array(ct.fit_transform(X))
	  print(X)
	  ```
		-
		- **Import**: Imports OneHotEncoder from Scikit-learn to convert categorical data into numerical format.
		- **Initialize ColumnTransformer**: Creates a ColumnTransformer named ct that:
		- Applies OneHotEncoder to column 0 (specified by [0]) of X, creating binary columns for each category.
		- Uses remainder='passthrough' to keep all other columns unchanged.
		- **Fit and Transform**: ct.fit_transform(X) applies one-hot encoding to column 0, transforms X, and converts the result to a NumPy array (since ColumnTransformer may output a sparse matrix).
		- **Update X**: Assigns the transformed data back to X, now with one-hot encoded columns for column 0 and original columns preserved.
		- **Print**: Displays the updated X with the encoded data.
-
- # **Label Encoder**
  collapsed:: true
	- ```python
	  from sklearn.preprocessing import LabelEncoder
	  le = LabelEncoder()
	  y = le.fit_transform(y)
	  print(y)
	  ```
		-
		- **Import**: Imports LabelEncoder from Scikit-learn to transform categorical labels into numerical values.
		- **Initialize LabelEncoder**: Creates a LabelEncoder object named le.
		- **Fit and Transform**: le.fit_transform(y) learns the unique categories in y and assigns each a numerical value, updating y with the encoded values.
		- **Print**: Displays the encoded y.
	-
-
- # **Feature Scaling**
  collapsed:: true
	- ```python
	  from sklearn.preprocessing import StandardScaler
	  sc = StandardScaler()
	  X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
	  X_test[:, 3:] = sc.transform(X_test[:, 3:])
	  print(X_train)
	  print(X_test)
	  ```
		-
		- **Import**: Imports StandardScaler from Scikit-learn for feature standardization.
		- **Initialize Scaler**: Creates a StandardScaler object named sc.
		- **Fit and Transform (Training Data)**: sc.fit_transform(X_train[:, 3:]) calculates the mean and standard deviation for columns 3 and beyond (index 3 to end) in X_train, then standardizes those columns by subtracting the mean and dividing by the standard deviation. The transformed values replace the original ones.
		- **Transform (Test Data)**: sc.transform(X_test[:, 3:]) applies the same mean and standard deviation (learned from X_train) to standardize columns 3 and beyond in X_test, ensuring consistency. The transformed values replace the original ones.
		- **Print**: Displays the updated X_train and X_test with standardized columns.
-