#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install sklearn')
get_ipython().system('pip install tensorflow')
get_ipython().system('pip install ')
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import tensorflow.keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


# In[5]:


df = pd.read_csv('Claims.csv')


# In[7]:


df.head()


# In[8]:


df.info()


# In[9]:


df.describe()


# In[10]:


df.shape


# In[11]:


df['Make'].unique()


# In[21]:


df.isnull().sum() 


# In[12]:


pd.set_option('display.max_columns', None)  # Set the option to display all columns
df.head()


# In[13]:


df['Sex'] = df['Sex'].map({'Male': 0, 'Female': 1})
df['MaritalStatus'] = df['MaritalStatus'].map({'Single': 0, 'Married': 1})


# In[14]:


df.head()


# In[15]:


df['MonthClaimed'].unique()


# In[16]:


df = df[df['MonthClaimed'] != '0']


# In[17]:


df['Month'] = pd.to_datetime(df['Month'], format='%b')
df['Month'] = df['Month'].dt.month


# In[18]:


df['DayOfWeek'].unique()


# In[19]:


day_mapping = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6
}
df['DayOfWeek'] = df['DayOfWeek'].map(day_mapping)


# In[20]:


df['MonthClaimed'] = pd.to_datetime(df['MonthClaimed'], format='%b')
df['MonthClaimed'] = df['MonthClaimed'].dt.month


# In[21]:


day_mapping = {
    'Monday': 0,
    'Tuesday': 1,
    'Wednesday': 2,
    'Thursday': 3,
    'Friday': 4,
    'Saturday': 5,
    'Sunday': 6
}
df['DayOfWeekClaimed'] = df['DayOfWeekClaimed'].map(day_mapping)


# In[22]:


df.head()


# In[23]:


df['Make'].unique()


# In[24]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Make'] = le.fit_transform(df['Make'])


# In[25]:


df.head()


# In[26]:


df['NumberOfCars'].unique()


# In[27]:


categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
print(categorical_columns)


# In[28]:


categorical_columns


# In[29]:


from sklearn.preprocessing import LabelEncoder

columns_to_encode = ['AccidentArea',
 'Fault',
 'PolicyType',
 'VehicleCategory',
 'VehiclePrice',
 'Days_Policy_Accident',
 'Days_Policy_Claim',
 'PastNumberOfClaims',
 'AgeOfVehicle',
 'AgeOfPolicyHolder',
 'PoliceReportFiled',
 'WitnessPresent',
 'AgentType',
 'NumberOfSuppliments',
 'AddressChange_Claim',
 'NumberOfCars',
 'BasePolicy']

le = LabelEncoder()

for column in columns_to_encode:
    df[column] = le.fit_transform(df[column])


# In[30]:


df.head()


# In[31]:


df.columns


# In[32]:


y = df['FraudFound_P']
X = df.drop('FraudFound_P', axis = 1)


# In[33]:


X.describe()


# In[34]:


print(df.shape)

print(df.dtypes)

print(df.isnull().sum())

print(df.describe())


# In[35]:


import matplotlib.pyplot as plt
import seaborn as sns

# Histogram of a numerical variable
sns.histplot(df['Age'])
plt.show()

# Box plot of a numerical variable
sns.boxplot(x='Make', y='Age', data=df)
plt.show()

# Scatter plot of two numerical variables
sns.scatterplot(x='Age', y='VehiclePrice', data=df)
plt.show()

# Bar plot of a categorical variable
sns.countplot(x='AccidentArea', data=df)
plt.show()


# In[40]:


df_if = pd.read_csv('C:/Users/skollu/Downloads/fraud_oracle.csv')


# In[41]:


import seaborn as sns
import matplotlib.pyplot as plt

# Calculate correlation matrix
corr_matrix = df_if.corr()

# Set up the matplotlib figure and axes
fig, ax = plt.subplots(figsize=(10, 8))

# Create a heatmap with custom color scheme
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr_matrix, annot=True, cmap=cmap, fmt=".2f", linewidths=0.5, ax=ax)

# Set the title and labels
ax.set_title("Correlation Matrix", fontsize=14)
ax.set_xlabel("Variables", fontsize=12)
ax.set_ylabel("Variables", fontsize=12)

# Rotate the x and y axis labels for better readability
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# Adjust the plot layout to prevent cutoff of labels
plt.tight_layout()

# Show the plot
plt.show()


# In[42]:


# Drop rows with NaN values
df.dropna(axis=0, inplace=True)


# In[43]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Separate the features and target variable
X = df.drop('FraudFound_P', axis=1)
y = df['FraudFound_P']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features separately for train and test sets
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[44]:


import tensorflow as tf
import tensorflow.keras


# In[ ]:


# Set random seed
tf.random.set_seed(42)

# Create the model
insurance_model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation = 'relu'),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(32, activation = 'relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
insurance_model_1.compile(loss = tf.keras.losses.mae,
                          optimizer = tf.keras.optimizers.Adam(lr = 0.1),
                          metrics = ['accuracy'])

# Fit the model
history = insurance_model_1.fit(X_train_scaled, y_train, epochs = 50)


# In[ ]:


insurance_model_1.evaluate(X_test, y_test)


# In[ ]:


y_pred = model.predict(X_test)
y_pred


# In[46]:



import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Print the confusion matrix
print("Confusion Matrix:")
print(cm)

# Plotting the confusion matrix
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Predicted 0', 'Predicted 1'])
plt.yticks(tick_marks, ['Actual 0', 'Actual 1'])
plt.xlabel('Predicted label')
plt.ylabel('Actual label')
plt.show()


# # Logistic Regression Model Implementation
# 

# In[47]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[48]:


X = df.drop('FraudFound_P', axis=1)
y = df['FraudFound_P']


# In[49]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[50]:


model = LogisticRegression()
model.fit(X_train, y_train)


# In[51]:


y_pred = model.predict(X_test)


# In[52]:


from sklearn import metrics
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred))


# # ISOLATION FOREST MODEL

# In[57]:


# Train the Isolation Forest model
model = IsolationForest(contamination=0.01)  # Adjust the contamination parameter based on the expected fraud rate
model.fit(X_train_scaled)

# Predict anomalies on the test set
anomaly_scores = model.decision_function(X_test_scaled)
predictions = model.predict(X_test_scaled)

# Identify the fraud cases in the test set
fraud_indices = [index for index, pred in enumerate(predictions) if pred == -1]
fraud_cases = X_test.iloc[fraud_indices]

# Print the fraud cases in the test set
fraud_cases
from sklearn.metrics import accuracy_score

# Assuming y_test contains the true labels for the test set
# Adjust y_test to reflect the actual true labels in your dataset

# Convert -1 to 0 for fraud cases in predictions
binary_predictions = [0 if pred == -1 else 1 for pred in predictions]

# Calculate accuracy
accuracy = accuracy_score(y_test, binary_predictions)

print(f"Accuracy: {accuracy}")


# from sklearn.metrics import confusion_matrix

# # Assuming 'y' is the true labels and 'predictions' is the predicted labels
# conf_matrix = confusion_matrix(y, predictions)
# print("Confusion Matrix:")
# print(conf_matrix)


# # DECISION TREE MODEL IMPLEMENTATION

# In[58]:


from sklearn.tree import DecisionTreeClassifier
X = df.drop('FraudFound_P', axis=1)
y = df['FraudFound_P']


# In[59]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[60]:


model = DecisionTreeClassifier()
model.fit(X_train, y_train)


# In[61]:


y_pred = model.predict(X_test)


# In[63]:


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

