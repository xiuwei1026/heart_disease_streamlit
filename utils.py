import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.express as px
import plotly.graph_objs as go

# import io
# import base64
# import os
# import json
# import pickle
# import uuid
# import re


from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@st.cache(persist=True)
def load_data():
    data = pd.read_csv('data/framingham_clean.csv',index_col=False)
    return data


@st.cache(persist=True)
def split(df):
    y = df.TenYearCHD
    x = df.drop(columns=['TenYearCHD'])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test


def plot_metrics(metrics_list, model, x_test, y_test, class_names):
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
        st.pyplot()

    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        plot_roc_curve(model, x_test, y_test)
        st.pyplot()

    if 'Precision-Recall Curve' in metrics_list:
        st.subheader("Precision-Recall Curve")
        plot_precision_recall_curve(model, x_test, y_test)
        st.pyplot()


def plot_displot(df, numeric_features):
    f, axes = plt.subplots(3, 2, figsize=(14,16))
    index1 = 0
    index2 = 0

    for col in numeric_features:
        sns.distplot(df[col], ax=axes[index1][index2]);
        index2 = index2+1
        if index2==2:
            index2 = 0
            index1 = index1+1
    st.pyplot()
    
def plot_barplot(df):
    data_cat= df.copy()
    data_cat['ageGroup'] = pd.cut(x=data_cat['age'], bins=[30, 39, 49, 59, 70], 
                          labels=['30-39 years', '40-49 years', '50-59 years', '60-70 years'])
    data_cat['gender'] = data_cat['gender'].map({0: 'female',1: 'male'})
    trace0 = go.Box(x=data_cat['ageGroup'],y=data_cat['sysBP'], 
                name='Systolic Blood Pressure', line=dict(color='steelblue'))
    trace1 = go.Box(x=data_cat['ageGroup'], y=data_cat['diaBP'], 
                name='Diastolic Blood Pressure', line=dict(color='goldenrod'))

    fig11 = go.Figure([trace0, trace1])
    fig11.update_layout(
    title={
        'text': "Blood Pressure by Age Group",
        'x':0.5,
        'xanchor': 'center'},
    xaxis={'categoryorder':'category ascending'},
    width = 1000,
    height = 500,
    margin=dict(
    l=50,
    r=50,
    b=100,
    t=100,
    pad=4
    ),)
    st.plotly_chart(fig11)

def plot_scatterplot(df):
    data_matrix = df.drop(['gender', 'education', 'prevalentStroke', 'prevalentHyp', 
                         'diabetes', 'BPMeds', 'currentSmoker'], axis=1)
    textd = ['Not at risk of CHD' if cl==0 else 'Risk of CHD' for cl in data_matrix['TenYearCHD']]

    fig12 = go.Figure(data=go.Splom(
                      dimensions=[dict(label='Age', values=data_matrix['age']),
                                  dict(label='Cigs per day', values=data_matrix['cigsPerDay']),
                                  dict(label='Total Cholesterol', values=data_matrix['totChol']),
                                  dict(label='Systolic BP', values=data_matrix['sysBP']),
                                  dict(label='Diastolic BP', values=data_matrix['diaBP']),
                                  dict(label='BMI', values=data_matrix['BMI']),
                                  dict(label='Heart Rate', values=data_matrix['heartRate']),
                                  dict(label='Glucose', values=data_matrix['glucose'])],
                      marker=dict(color=data_matrix['TenYearCHD'],
                                  size=5,
                                  colorscale='Bluered',
                                  line=dict(width=0.5,
                                            color='rgb(230,230,230)')),
                      text=textd,
                      diagonal=dict(visible=False)))

    title = "Scatterplot Matrix (SPLOM) for Framingham Heart Study Dataset"
    fig12.update_layout(title=title,
                      dragmode='select',
                      width=1000,
                      height=1000,
                      hovermode='closest')
    st.plotly_chart(fig12)
    
def download_button(object_to_download, download_filename, button_text, pickle_it=False):
    """
    Generates a link to download the given object_to_download.
    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.
    Returns:
    -------
    (str): the anchor tag to download object_to_download
    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')
    """
    if pickle_it:
        try:
            object_to_download = pickle.dumps(object_to_download)
        except pickle.PicklingError as e:
            st.write(e)
            return None

    else:
        if isinstance(object_to_download, bytes):
            pass

        elif isinstance(object_to_download, pd.DataFrame):
            #object_to_download = object_to_download.to_csv(index=False)
            towrite = io.BytesIO()
            object_to_download = object_to_download.to_excel(towrite, encoding='utf-8', index=False, header=True)
            towrite.seek(0)

        # Try JSON encode for everything else
        else:
            object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError as e:
        b64 = base64.b64encode(towrite.read()).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = custom_css + f'<a download="{download_filename}" id="{button_id}" href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}">{button_text}</a><br></br>'

    return dl_link