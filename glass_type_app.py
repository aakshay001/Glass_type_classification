# Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'glass_type_app.py'.
# You have already created this ML model in ones of the previous classes.

# Importing the necessary Python modules.
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

# ML classifier Python modules
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
 
# Loading the dataset.
@st.cache()
def load_data():
    file_path = "glass-types.csv"
    df = pd.read_csv(file_path, header = None)
    # Dropping the 0th column as it contains only the serial numbers.
    df.drop(columns = 0, inplace = True)
    column_headers = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'GlassType']
    columns_dict = {}
    # Renaming columns with suitable column headers.
    for i in df.columns:
        columns_dict[i] = column_headers[i - 1]
        # Rename the columns.
        df.rename(columns_dict, axis = 1, inplace = True)
    return df

glass_df = load_data() 

# Creating the features data-frame holding all the columns except the last column.
X = glass_df.iloc[:, :-1]

# Creating the target series that holds last column.
y = glass_df['GlassType']

# Spliting the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# Create a function that accepts an ML model object say 'model' and the nine features as inputs 
# and returns the glass type.
@st.cache()
def prediction(model, ri, na, mg, al, si, k, ca, ba, fe):
    glass_type = model.predict([[ri, na, mg, al, si, k, ca, ba, fe]])
    glass_type = glass_type[0]
    if glass_type == 1:
        return "building windows float processed".upper()
    elif glass_type == 2:
        return "building windows non float processed".upper()
    elif glass_type == 3:
        return "vehicle windows float processed".upper()
    elif glass_type == 4:
        return "vehicle windows non float processed".upper()
    elif glass_type == 5:
        return "containers".upper()
    elif glass_type == 6:
        return "tableware".upper()
    else:
        return "headlamps".upper()


# Add title on the main page and in the sidebar.
st.title("Glass Type prediction Web app")
st.sidebar.title("Glass Type prediction Web app")


# Using if statement, display raw data on the click of the checkbox.
if st.sidebar.checkbox("Show raw data"):
    st.subheader("Glass Type Data set")
    st.dataframe(glass_df)

# S1.1: Add a multiselect widget to allow the user to select multiple visualisation.

st.sidebar.subheader("Visialisation Selector")
plot_list = st.sidebar.multiselect("Select the charts/Plots",("Correlation Heatmap","Line Chart","Area Chart","Count Plot","Pie Chart","Box Plot"))


# S1.2: Display Streamlit native line chart and area chart
if "Line Chart" in plot_list:
  st.subheader("Line Chart")
  st.line_chart(glass_df)

if "Area Chart" in plot_list:
  st.subheader("Area Chart")
  st.area_chart(glass_df)
  

# S1.3: Display the plots when the user selects them using multiselect widget.
st.set_option('deprecation.showPyplotGlobalUse', False)

if "Correlation Heatmap" in plot_list:
  st.subheader("Correlation Heatmap")
  plt.figure(figsize=(9,5))
  sns.heatmap(glass_df.corr(),annot = True)
  st.pyplot()

if "Count Plot" in plot_list:
  st.subheader("Count Plot")
  plt.figure(figsize=(9,5))
  sns.countplot("GlassType",data = glass_df,)
  st.pyplot()  

if "Pie Chart" in plot_list:
  st.subheader("Pie Chart")
  plt.pie(glass_df["GlassType"].value_counts(),labels = glass_df["GlassType"].value_counts().index, autopct = '%1.2f%%', startangle = 30, explode = np.linspace(.06, .12, 6))
  st.pyplot()

  

# S1.4: Display box plot using matplotlib module and 'st.pyplot()'
if "Box Plot" in plot_list:
  st.subheader("Box Plot")
  column = st.sidebar.selectbox("Select Column For Box Plot",('RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'))
  sns.boxplot(glass_df[column])
  st.pyplot()

# S2.1: Add 9 slider widgets for accepting user input for 9 features.
st.sidebar.subheader("Select your Values")
ri = st.sidebar.slider("RI",float(glass_df["RI"].min()),float(glass_df["RI"].max()))

na = st.sidebar.slider("Na",float(glass_df["RI"].min()),float(glass_df["Na"].max()))

mg = st.sidebar.slider("Mg",float(glass_df["Mg"].min()),float(glass_df["Mg"].max()))

al = st.sidebar.slider("Al",float(glass_df["Al"].min()),float(glass_df["Al"].max()))

si = st.sidebar.slider("Si",float(glass_df["Si"].min()),float(glass_df["Si"].max()))

k = st.sidebar.slider("K",float(glass_df["K"].min()),float(glass_df["K"].max()))

ca = st.sidebar.slider("Ca",float(glass_df["Ca"].min()),float(glass_df["Ca"].max()))

ba = st.sidebar.slider("Ba",float(glass_df["Ba"].min()),float(glass_df["Ba"].max()))

fe = st.sidebar.slider("Fe",float(glass_df["Fe"].min()),float(glass_df["Fe"].max()))


# S3.1: Add a subheader and multiselect widget.
st.sidebar.subheader("Choose Classifier")
classifier = st.sidebar.selectbox("Classifier",('Logistic Regression','Random Forest Classifier',"Support Vector Machine"))


# S4.1: Implement SVM with hyperparameter tuning
from sklearn.metrics import plot_confusion_matrix
if classifier == "Support Vector Machine":
  st.sidebar.subheader("Model Hyperparameter")
  c = st.sidebar.number_input("C(Error rate)",1,100,step = 1)
  kernel = st.sidebar.radio("Kernel",("linear","rbf","poly"))
  gamma = st.sidebar.number_input("Gamma",1,100,step=1)
  if st.sidebar.button("Classify"):
    st.subheader("Support Vector Machine")
    svc_model = SVC(C=c,kernel=kernel,gamma=gamma)
    svc_model.fit(X_train,y_train)
    y_pred = svc_model.predict(X_test)
    score = svc_model.score(X_train,y_train)
    glass_type = prediction(svc_model,ri,na,mg,al,si,k,ca,ba,fe)
    st.write("The Type of glass predict is: ",glass_type)
    st.write("Accuracy: ",score.round(2))
    plot_confusion_matrix(svc_model,X_test,y_test)
    st.pyplot()

# S5.1: ImplementRandom Forest Classifier with hyperparameter tuning.
if classifier == "Random Forest Classifier":
  st.sidebar.subheader("Model Hyperparameter")
  n_estimator = st.sidebar.number_input("Number Of Decision Trees",100,10000,step = 10)
  max_depth = st.sidebar.number_input("Maximum Depth of Tree",1,100,step=1)
  if st.sidebar.button("Classify"):
    st.subheader("Random Forest Classifier")
    rfc = RandomForestClassifier(n_estimators =n_estimator,max_depth=max_depth)
    rfc.fit(X_train,y_train)
    y_pred = rfc.predict(X_test)
    score = rfc.score(X_train,y_train)
    glass_type = prediction(rfc,ri,na,mg,al,si,k,ca,ba,fe)
    st.write("The Type of glass predict is: ",glass_type)
    st.write("Accuracy: ",score.round(2))
    plot_confusion_matrix(rfc,X_test,y_test)
    st.pyplot() 

if classifier == "Logistic Regression":
    if st.sidebar.button("Classify"):
        st.subheader("Logistic Regression")
        log_reg = LogisticRegression()
        log_reg.fit(X_train,y_train)
        y_pred = log_reg.predict(X_test)
        score = log_reg.score(X_train,y_train)
        glass_type = prediction(log_reg,ri,na,mg,al,si,k,ca,ba,fe)
        st.write("The Type of glass predict is: ",glass_type)
        st.write("Accuracy: ",score.round(2))
        plot_confusion_matrix(log_reg,X_test,y_test)
        st.pyplot() 

