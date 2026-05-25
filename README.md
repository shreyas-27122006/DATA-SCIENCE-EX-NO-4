# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Scaling for the feature in the data set.

STEP 4:Apply Feature Selection for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1

2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.

3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.

4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.

The feature selection techniques used are:

1.Filter Method

2.Wrapper Method

3.Embedded Method

# CODING AND OUTPUT:
~~~
        FEATURE SCALING
import pandas as pd
from scipy import stats
import numpy as np
~~~
~~~
df=pd.read_csv("/content/bmi.csv")
df.head()
~~~
~~~
<img width="1037" height="536" alt="image" src="https://github.com/user-attachments/assets/b7c41cd3-cdaf-412c-911d-6f9b4c2046b4" />
~~~
~~~
df_null_sum=df.isnull().sum()
df_null_sum
~~~

~~~
<img width="906" height="454" alt="image" src="https://github.com/user-attachments/assets/8277e5ec-bc64-4908-9bd9-596ff651f303" />
~~~

~~~
df.dropna()
~~~

~~~
<img width="1044" height="470" alt="image" src="https://github.com/user-attachments/assets/9180e469-3627-4876-8f52-47c05cb414d6" />
~~~

~~~
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
# This is typically used in feature scaling,
#particularly max-abs scaling, which is useful
#when you want to scale data to the range [-1, 1]
#while maintaining sparsity (often used with sparse data).
~~~

~~~
<img width="1031" height="482" alt="image" src="https://github.com/user-attachments/assets/8ddb05d2-68fe-41fe-8488-09eaf3b96c1a" />
~~~

~~~
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("/content/bmi.csv")
df1.head()
~~~

~~~
<img width="1040" height="374" alt="image" src="https://github.com/user-attachments/assets/8dd0b10f-6f9f-4ee0-b2ac-8c0d8b6ed065" />
~~~

~~~
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
~~~

~~~
<img width="1041" height="480" alt="image" src="https://github.com/user-attachments/assets/b1c4d2f5-f2d6-467c-bb90-bd2f4bac0a69" />
~~~

~~~
#MIN-MAX SCALING:
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
~~~

~~~
<img width="1043" height="555" alt="image" src="https://github.com/user-attachments/assets/39f76c07-469e-4d69-8ee6-d08c4e0bfe30" />
~~~

~~~
#MAXIMUM ABSOLUTE SCALING:
from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("/content/bmi.csv")
df3.head()
~~~

~~~
<img width="1040" height="562" alt="image" src="https://github.com/user-attachments/assets/a6e2af0f-6af5-4dc5-bdf6-e97ea0e40e0a" />
~~~

~~~
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
~~~

~~~
<img width="1028" height="531" alt="image" src="https://github.com/user-attachments/assets/552378fd-5004-4b17-8ec1-21cc50885a77" />
~~~

~~~
#ROBUST SCALING
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3.head()
~~~

~~~
<img width="1031" height="559" alt="image" src="https://github.com/user-attachments/assets/04d00b3d-abcd-41b2-8f09-4007185c7740" />
~~~

~~~
#FEATURE SELECTION:
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
~~~

~~~
<img width="1044" height="552" alt="image" src="https://github.com/user-attachments/assets/fdcd39c8-c12f-4e27-a930-2b2a9c1fbc32" />
~~~

~~~
df_null_sum=df.isnull().sum()
df_null_sum
~~~

~~~
<img width="900" height="804" alt="image" src="https://github.com/user-attachments/assets/06399b2e-0584-44cd-97ec-0b36c7bba60c" />
~~~

~~~
# Chi_Square
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
~~~
~~~
# Chi_Square
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
#In feature selection, converting columns to categorical helps certain algorithms
# (like decision trees or chi-square tests) correctly understand and
# process non-numeric features. It ensures the model treats these columns as categories,
# not as continuous numerical values.
df[categorical_columns]
~~~

~~~
<img width="1040" height="529" alt="image" src="https://github.com/user-attachments/assets/24299007-6ca0-41c5-ac73-561c2a89a3ff" />
~~~

~~~
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
##This code replaces each categorical column in the DataFrame with numbers that represent the categories.
df[categorical_columns]
~~~

~~~
<img width="1037" height="615" alt="image" src="https://github.com/user-attachments/assets/ff8b7c28-5cd2-4cec-af58-c2f1edbc2e41" />
~~~

~~~
X = df.drop(columns=['SalStat'])
y = df['SalStat']
#X contains all columns except 'SalStat' — these are the input features used to predict something.
#y contains only the 'SalStat' column — this is the target variable you want to predict
~~~
~~~
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
~~~

~~~
<img width="1041" height="257" alt="image" src="https://github.com/user-attachments/assets/90d5a714-568b-4616-9c14-484ec6dd3ee7" />
~~~

~~~
y_pred = rf.predict(X_test)
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
~~~

~~~
<img width="1029" height="644" alt="image" src="https://github.com/user-attachments/assets/fbba6dc6-1018-4982-8ad6-acc0fe96a4b2" />
~~~

~~~
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
~~~

~~~
<img width="1041" height="544" alt="image" src="https://github.com/user-attachments/assets/9d01b10a-e95e-4640-ac3b-fd61c59c2420" />
~~~

~~~
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
~~~

~~~
<img width="1023" height="554" alt="image" src="https://github.com/user-attachments/assets/bfbfd04e-b1a6-40c8-8c4d-a190830c11a6" />
~~~

~~~

X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)
~~~

~~~
<img width="1029" height="339" alt="image" src="https://github.com/user-attachments/assets/c9638149-f57c-4241-ba11-30fc15378ddc" />
~~~

~~~

import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split # Importing the missing function
from sklearn.ensemble import RandomForestClassifier
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
~~~

~~~
<img width="1027" height="362" alt="image" src="https://github.com/user-attachments/assets/68eb9d84-0d3d-4140-b4c9-d1234a412a41" />
~~~

~~~
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
~~~

~~~
<img width="790" height="152" alt="image" src="https://github.com/user-attachments/assets/0bffed44-0102-4de1-94c2-70f058f297c4" />
~~~

~~~
!pip install skfeature-chappers
~~~

~~~
<img width="1034" height="278" alt="image" src="https://github.com/user-attachments/assets/80faaa6a-f1a9-4606-87bd-9c688a51608a" />
~~~

~~~
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
~~~
~~~
categorical_columns = [
'JobType',
'EdType',
'maritalstatus',
'occupation',
'relationship',
'race',
'gender',
'nativecountry'
]
df[categorical_columns] = df[categorical_columns].astype('category')
~~~
~~~
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
# @title
df[categorical_columns]
~~~

~~~
<img width="1035" height="592" alt="image" src="https://github.com/user-attachments/assets/f0208086-2e74-40b3-b2b4-955888e78e1a" />
~~~

~~~
X = df.drop(columns=['SalStat'])
y = df['SalStat']
~~~
~~~
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
X_anova = selector_anova.fit_transform(X, y)
~~~
~~~
selected_features_anova = X.columns[selector_anova.get_support()]
~~~
~~~
print("\nSelected features using ANOVA:")
print(selected_features_anova)
~~~

~~~
<img width="1020" height="148" alt="image" src="https://github.com/user-attachments/assets/387b1914-957e-44c2-b132-5cb3ec280435" />
~~~

~~~
# Wrapper Method
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
df=pd.read_csv("/content/income(1) (1).csv")
# List of categorical columns
categorical_columns = [
'JobType',
'EdType',
'maritalstatus',
'occupation',
'relationship',
'race',
'gender',
'nativecountry'
]
~~~
~~~
# Convert the categorical columns to category dtype
df[categorical_columns] = df[categorical_columns].astype('category')
~~~
~~~
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
~~~
~~~
df[categorical_columns]
~~~

~~~
<img width="1038" height="464" alt="image" src="https://github.com/user-attachments/assets/89bd3845-0af4-4a0a-b2b7-f5252d78e5f4" />
~~~

~~~
X = df.drop(columns=['SalStat'])
y = df['SalStat']
~~~
~~~
logreg = LogisticRegression()
~~~
~~~
n_features_to_select =6
~~~
~~~
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
~~~

~~~
<img width="1034" height="439" alt="image" src="https://github.com/user-attachments/assets/bd6fb838-687d-4d6b-9c4d-d0bf38981b91" />
~~~

~~~
<img width="1042" height="458" alt="image" src="https://github.com/user-attachments/assets/2dbee410-2dd7-467a-b590-414dcfb519ca" />
~~~
# RESULT:
The given code is successfully executed.
