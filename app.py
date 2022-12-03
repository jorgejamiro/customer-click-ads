import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
from io import BytesIO

import i18n

from utils import fig_confustion_matrix, fig_prediction_areas, fig_histogram
from languages import language_init


@st.cache
def load_data():
    df = pd.read_csv('Facebook_Ads_2.csv', encoding='ISO-8859-1')

    return df


def load_model(model):
    with open(model, 'rb') as file:
        data = pickle.load(file)

    return data


def show_data(df):
    fig1 = plt.figure(figsize=(10, 4))
    sns.scatterplot(x=df['Time Spent on Site'], y=df['Salary'], hue=df['Clicked'], legend='full') 
    plt.title(i18n.t('title_graph_1'))
    plt.xlabel(i18n.t('Time Spent on Site'))
    plt.ylabel(i18n.t('Salary'))
    plt.legend(title='Click')
    
    st.pyplot(fig1)
    st.markdown(i18n.t('comment_graph_1'))

    fig2_opt = st.selectbox('', options=('cs', 'ctss'), format_func=lambda x: i18n.t(f'{x}'))

    fig2 = plt.figure(figsize=(7, 7))
    if (fig2_opt == 'cs'):
        sns.boxplot(x='Clicked', y='Salary', data=df).set_title(i18n.t('title_graph_2_1'))
        plt.xlabel('Click')
        plt.ylabel(i18n.t('Salary'))
    else:
        sns.boxplot(x='Clicked', y='Time Spent on Site', data=df).set_title(i18n.t('title_graph_2_2'))
        plt.xlabel('Click')
        plt.ylabel(i18n.t('Time Spent on Site'))

    buf = BytesIO()
    fig2.savefig(buf, format="png")
    _, col2, _ = st.columns((1, 4, 1))
    col2.image(buf)
    st.markdown(i18n.t('comment_graph_2'))

    fig3_opt = st.selectbox('', ('Salary', 'Time Spent on Site'), format_func=lambda x: i18n.t(f'{x}'))

    fig3 = fig_histogram(df, fig3_opt, 8, 6)

    fig3.savefig(buf, format="png")
    _, col2, _ = st.columns((1, 8, 1))
    col2.image(buf)

    st.markdown(i18n.t('comment_1_graph_3'))
    st.markdown(i18n.t('comment_2_graph_3'))


def show_predict():
    data = load_model('./saved_model_p1.pk1')
    classifier = data['model']
    x_train = data['x_train']
    x_test = data['x_test']
    y_train = data['y_train']
    y_test = data['y_test']

    y_predict_train = classifier.predict(x_train)
    y_predict_test = classifier.predict(x_test)

    fig_opt = st.selectbox('', ('Train Data', 'Test Data'), format_func=lambda x: i18n.t(f'{x}'))

    col1, col2 = st.columns((1.5,2))
    if fig_opt == 'Train Data':
        col1.pyplot(fig_confustion_matrix(y_train, y_predict_train, 6, 4))
        col2.pyplot(fig_prediction_areas(x_train, y_train, classifier, 8, 4))
    else:
        col1.pyplot(fig_confustion_matrix(y_test, y_predict_test, 6, 4))
        col2.pyplot(fig_prediction_areas(x_test, y_test, classifier, 8, 4))

    st.markdown(i18n.t('description_graphs'), unsafe_allow_html=True)
    st.markdown(i18n.t('comment_last_1'))
    st.markdown('\n')
    st.markdown(i18n.t('comment_last_2'))


language_init()

st.markdown(i18n.t('title'), unsafe_allow_html=True)
st.markdown('\n')
st.markdown(i18n.t('intro_1'))
st.markdown(i18n.t('intro_2'))
st.markdown('\n')

df_training_set = load_data()
show_data(df_training_set)
show_predict()








