import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from preprocess_data import clean_data, split_data
from analyze_charts import print_models

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

filename = 'datasets/water_potability.csv'

@st.cache(allow_output_mutation=True)
def load_data():
    data = pd.read_csv(filename)
    return data

st.header('ML. Анализ датасета, сравнение моделей машинного обучений и подробный анализ метода ближайших соседей')

data_load_state = st.text('Загрузка данных...')
data = load_data()
data_load_state.text('Датасет ' + filename + ' был загружен')
data_len = data.shape[0]

if st.checkbox('Показать первые 5 значений'):
  st.write(data.head())

if st.checkbox('Показать все данные'):
    st.subheader('Данные')
    st.write(data)

if st.checkbox('Показать cкрипичные диаграммы для числовых колонок'):
  for col in data.columns:
      fig1 = plt.figure(figsize=(10,6))
      ax = sns.violinplot(x=data[col])
      st.pyplot(fig1)

if st.checkbox('Показать корреляционную матрицу'):
    fig1, ax = plt.subplots(figsize=(10,5))
    sns.heatmap(data.corr(), annot=True, fmt='.2f')
    st.pyplot(fig1)


st.subheader('Очистка и деление датасета')
data_X, data_y = clean_data(data)
st.write('Датасет очищен!')
X_train, X_test, y_train, y_test = split_data(data_X, data_y)
st.write('Датасет разделен!')

if st.checkbox('Показать первые 5 значений очищенного датасета'):
  st.write(data_X.head())


st.subheader('Анализ метода ближайших соседей')
if st.checkbox('Испытать метод'):
  cv_slider = st.slider('Количество фолдов:', min_value=3, max_value=10, value=5, step=1)
  rows_in_one_fold = int(data_len / cv_slider)
  allowed_knn = int(rows_in_one_fold * (cv_slider-1))
  st.write('Количество строк в наборе данных - {}'.format(data_len))
  st.write('Максимальное допустимое количество ближайших соседей с учетом выбранного количества фолдов - {}'.format(allowed_knn))

  cv_knn = st.slider('Количество ближайших соседей:', min_value=1, max_value=allowed_knn, value=5, step=1)

  scores = cross_val_score(KNeighborsClassifier(n_neighbors=cv_knn), 
      data_X, data_y, scoring='accuracy', cv=cv_slider)

  st.subheader('Оценка качества модели')
  st.write('Значения accuracy для отдельных фолдов')
  st.bar_chart(scores)
  st.write('Усредненное значение accuracy по всем фолдам - {}'.format(np.mean(scores)))

st.subheader('Сравнений разных моделей машинного обучения')
if st.checkbox('Сравнить модели'):
  st.sidebar.header('Выбор моделей')
  models_list = ['LogR', 'KNN_5', 'SVC', 'Tree', 'RF', 'GB']
  class_models = {'LogR': LogisticRegression(), 
                'KNN_5':KNeighborsClassifier(n_neighbors=5),
                'SVC':SVC(probability=True),
                'Tree':DecisionTreeClassifier(),
                'RF':RandomForestClassifier(),
                'GB':GradientBoostingClassifier()}

  models_select = st.sidebar.multiselect('Выберите модели', models_list)

  st.header('Оценка качества моделей')
  print_models(models_select, X_train, X_test, y_train, y_test, class_models)



