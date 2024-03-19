import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline

from google.colab import drive
drive.mount("./drive", force_remount=True)
path_prefix = "./drive/My Drive" 

filename = "KaggleV2-May-2016.csv"
df = pd.read_csv(join(path_prefix, filename))
df.head()

continuous_attributes=["PatientId", "AppointmentID", "ScheduledDay", "AppointmentDay", "Age", "Scolarship", "Hipertension", "Diabetes", "Alcoholism", 'Handcap', 'SMS_received']
categorical_attributes=["Gender", "Neighbourhood", "No-show"]

df = df[['Age', 'Gender']]
grouped_data = df.groupby('Gender').mean()
ages = grouped_data['Age']
plt.bar(x=['Female', 'Male'], height = ages)
plt.xlabel('Gender')
plt.ylabel('Mean Age')
plt.show()

df = df[['Age', 'Hipertension']]
grouped_data = df.groupby('Age').mean()
hipertension = grouped_data['Hipertension']
plt.bar(x=grouped_data.index, height=hipertension)
plt.xlabel('Age')
plt.ylabel('Mean Hypertension')
plt.show()

df = df[['Age', 'Diabetes']]
grouped_data = df.groupby('Age').sum()
grouped_data = grouped_data.reset_index()
diabetes = grouped_data['Diabetes']
plt.bar(x=grouped_data['Age'], height=diabetes)
plt.xlabel('Age')
plt.ylabel('Number of People with Diabetes')
plt.show()

df = df[['Diabetes', 'Hipertension']]
total = df['Diabetes'].sum() + df['Hipertension'].sum()
diabetes_percentage = df['Diabetes'].sum() / total * 100
hypertension_percentage = df['Hipertension'].sum() / total * 100
plt.pie(x=[diabetes_percentage, hypertension_percentage], labels=['Diabetes', 'Hipertension'], autopct='%1.1f%%')
plt.title('Diabetes and Hipertension Ratio')
plt.show()

labels = ['18-25', '25-40', '40-65']
alcoholism_rates = [20, 15, 10]
plt.pie(alcoholism_rates, labels=labels, autopct='%1.1f%%')
plt.axis('equal')
plt.title('Alcoholism rates by age group')
plt.show()


df = df[['Alcoholism', 'Hipertension']]
df = df.dropna()
total = df['Alcoholism'].sum() + df['Hipertension'].sum()
alcoholism_percentage = df['Alcoholism'].sum() / total * 100
hipertension_percentage = df['Hipertension'].sum() / total * 100
plt.pie(x=[alcoholism_percentage, hipertension_percentage], labels=['Alcoholism', 'Hipertension'], autopct='%1.1f%%')
plt.title('Alcoholism and Hipertension Ratio')
plt.show()


import matplotlib.pyplot as plt
df = df[['Alcoholism', 'Diabetes']]
df = df.dropna()
total = df['Alcoholism'].sum() + df['Diabetes'].sum()
alcoholism_percentage = df['Alcoholism'].sum() / total * 100
diabetes_percentage = df['Diabetes'].sum() / total * 100
plt.pie(x=[alcoholism_percentage, diabetes_percentage], labels=['Alcoholism', 'Diabetes'], autopct='%1.1f%%')
plt.title('Alcoholism and Diabetes Ratio')
plt.show()


features=df[['Age','Diabetes']]
labels=pd.to_numeric(df['Hipertension'])


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
# Split the data into a training set and a test set

# 80% train, 10% validation, 10% test
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.50, random_state=42)


from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

#Decision Tree Training
model_dt = tree.DecisionTreeClassifier(random_state=42) #Create decision tree classifier object
model_dt.fit(X_train, y_train) #train the classifier using the training data

#Random Forest Training(In new version default of estimators will be 100)
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve


rf_predictions_val = model_rf.predict(X_val)
rf_acc_val = accuracy_score(y_val, rf_predictions_val)

print("Random Forest Validation Accuracy:"+str(rf_acc_val))
