import streamlit as st
import numpy as np
import utils
import pandas as pd


from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score

#from SessionState import get


st.set_option('deprecation.showPyplotGlobalUse', False)


def main():
    st.title("Heart Disease Prediction by Xiupeng")
    st.sidebar.title("Heart Disease Prediction ")
    st.markdown("Do you have heart disease? ❤️")
    st.sidebar.markdown("Do you have heart disease? ❤️")


    df = utils.load_data()
    x_train, x_test, y_train, y_test = utils.split(df)
    
    # Data analysis
    class_names = ["Low Risk", "High Risk"]
    numeric_features = ['cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'glucose']
    
    st.sidebar.subheader("Data Visualization and Analytics")
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Heart Disease Data Set (Prediction)")
        st.write(df)
        filename = 'heart_data.xlsx'
        download_button_str = utils.download_button(df, filename, f'Click here to download {filename}', pickle_it=False)
        st.markdown(download_button_str, unsafe_allow_html=True)

    if st.sidebar.checkbox("Show distribution plot", False):
        st.subheader("Heart Disease Data Set Distribution plot")
        utils.plot_displot(df, numeric_features)  

    if st.sidebar.checkbox("Show box plot", False):
        st.subheader("Blood Pressure by Age Group")
        utils.plot_barplot(df)
        
    if st.sidebar.checkbox("Show scatter plot", False):
        st.subheader("Scatterplot Matrix")
        utils.plot_scatterplot(df)        
        
    # Machine learning
    st.sidebar.subheader("Machine Learning Prediction")
    st.sidebar.subheader("Choose Classifier")

    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression",
                                                     "Random Forest Classification"))


    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect("What matrix to plot?", ("Confusion Matrix", "ROC Curve",
                                                                  "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            utils.plot_metrics(metrics, model, x_test, y_test, class_names)


    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='Lr')
        max_iter = st.sidebar.slider("Maximum no. of iterations", 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect("What matrix to plot?", ("Confusion Matrix", "ROC Curve",
                                                                  "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            utils.plot_metrics(metrics, model, x_test, y_test, class_names)


    if classifier == 'Random Forest Classification':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("This is the number of trees in the forest", 100, 5000, step=10,
                                               key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 100, step=2, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ("True", "False"), key='bootstrap')
        metrics = st.sidebar.multiselect("What matrix to plot?", ("Confusion Matrix", "ROC Curve",
                                                                  "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            utils.plot_metrics(metrics, model, x_test, y_test, class_names)


if __name__ == '__main__':
    # session_state = get(password='')

    # if session_state.password != 'dss123':
        # pwd_placeholder = st.sidebar.empty()
        # pwd = pwd_placeholder.text_input("Password:", value="", type="password")
        # session_state.password = pwd
        # if session_state.password == 'dss123':
            # pwd_placeholder.empty()
            # main()
        # else:
            # st.error("Please type the correct password")
    # else:
        # main()
    main()