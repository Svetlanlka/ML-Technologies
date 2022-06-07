from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import streamlit as st

strategies=['mean', 'median', 'most_frequent']

# импьютация нужной колонки с помощью нужной стратегии
def func_impute_col(dataset, column, strategy_param):
    temp_data = dataset[[column]]
    
    imp_num = SimpleImputer(strategy=strategy_param)
    data_num_imp = imp_num.fit_transform(temp_data)
    
    return data_num_imp

@st.cache
def clean_data(data):
  # замена медианой pH
  col_imp = func_impute_col(data, 'ph', strategies[1])
  data[['ph']] = col_imp

  # замена медианой Sulfate
  col_imp = func_impute_col(data, 'Sulfate', strategies[1])
  data[['Sulfate']] = col_imp

  # замена медианой Trihalomethanes
  col_imp = func_impute_col(data, 'Trihalomethanes', strategies[1])
  data[['Trihalomethanes']] = col_imp

  # уберем столбцы, слабо коррелирующие с целевым признаком
  data_clean = data
  data_clean = data_clean.drop(columns = ['ph'], axis = 1)
  data_clean = data_clean.drop(columns = ['Conductivity'], axis = 1)
  data_clean = data_clean.drop(columns = ['Trihalomethanes'], axis = 1)
  data_clean = data_clean.drop(columns = ['Turbidity'], axis = 1)
  data_clean = data_clean.drop(columns = ['Hardness'], axis = 1)
  data_clean2 = data_clean.drop(columns = ['Potability'], axis = 1)

  return data_clean2, data[['Potability']]

@st.cache
def split_data(data, target):
  x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.2,random_state=1)
  return x_train,x_test,y_train,y_test