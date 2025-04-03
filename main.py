import streamlit as st
import pandas as pd
from machine_learning_framework.preprocessing.analyzer import Analyzer
from machine_learning_framework.model.classifier import LogisticRegressionClassifier, RandomForestClassification, SVMClassifier, NNClassifier, LightGBMClassifier
from machine_learning_framework.model.regressor import LinearRegressor, ElasticNetRegressor, RandomForestRegression, NNRegressor, LightGBMRegressor
from machine_learning_framework.model.clustering import KMeansClustering, AgglomerativeClusteringModel, DBSCANClustering, GaussianMixtureClustering

def main():
    st.title("Machine Learning Framework App")
    st.write("A generic machine-learning framework for convenient data analysis.")

    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("## Uploaded Data")
        st.write(df)

        # Initialize Analyzer
        if 'analyzer' not in st.session_state:
            analyzer = Analyzer(df)
            st.session_state.analyzer = analyzer

        # Display basic data analysis options
        st.write("## Data Analysis")
        if st.button("Describe Data"):
            st.write(st.session_state.analyzer.describe())

        if st.button("Drop Missing Data"):
            analyzer = st.session_state.analyzer
            analyzer.drop_missing_data()
            st.write("Missing data dropped.")
            st.write(analyzer.df)
            st.session_state.analyzer = analyzer

        cols_to_drop = st.multiselect("Select columns to drop", st.session_state.analyzer.df.columns)
        if st.button("Drop Columns"):
            analyzer =  st.session_state.analyzer
            analyzer.drop_columns(cols_to_drop)
            st.write("Columns dropped.")
            st.write(analyzer.df)
            st.session_state.analyzer = analyzer

        # Encoding options
        st.write("## One Hot Data Encoding")
        cols_to_encode = st.multiselect("Select columns to one-hot encode", st.session_state.analyzer.df.columns)
        if st.button("One-Hot Encode"):
            analyzer = st.session_state.analyzer
            analyzer.encode_one_hot(cols_to_encode)
            st.write("Columns one-hot encoded.")
            st.write(analyzer.df)
            st.session_state.analyzer = analyzer

        st.write("## Label Encoding")
        cols_to_encode = st.multiselect("Select columns to label encode", st.session_state.analyzer.df.columns)
        if st.button("Label Encode"):
            analyzer = st.session_state.analyzer
            analyzer.encode_label(cols_to_encode)
            st.write("Columns label encoded.")
            st.write(analyzer.df)
            st.session_state.analyzer = analyzer

        st.write("## Normalize Data")
        cols_to_encode = st.multiselect("Select columns to normalize", st.session_state.analyzer.df.columns)
        if st.button("Normalize Data"):
            analyzer = st.session_state.analyzer
            analyzer.encode_standard(cols_to_encode)
            st.write("Columns normalized.")
            st.write(analyzer.df)
            st.session_state.analyzer = analyzer

        # Classification options
        st.write("## Classification")
        classifier_options = ["Logistic Regression", "Random Forest", "SVM", "Neural Network", "LightGBM"]
        classifier_choice = st.selectbox("Choose a classifier", classifier_options)
        classifier = None

        if classifier_choice == "Logistic Regression":
            classifier = LogisticRegressionClassifier()
        elif classifier_choice == "Random Forest":
            classifier = RandomForestClassification()
        elif classifier_choice == "SVM":
            classifier = SVMClassifier()
        elif classifier_choice == "Neural Network":
            classifier = NNClassifier()
        elif classifier_choice == "LightGBM":
            classifier = LightGBMClassifier()

        target_column = st.selectbox("Select target column", st.session_state.analyzer.df.columns)
        if classifier and st.button("Train Classifier"):
            X = st.session_state.analyzer.df.drop(target_column, axis=1)
            y = st.session_state.analyzer.df[target_column]
            classifier.fit(X.values, y.values)
            st.write("Classifier trained.")
            st.write(classifier.get_classification_report())

        # Regression options
        st.write("## Regression")
        regressor_options = ["Linear Regression", "ElasticNet", "Random Forest", "Neural Network", "LightGBM"]
        regressor_choice = st.selectbox("Choose a regressor", regressor_options)
        regressor = None

        if regressor_choice == "Linear Regression":
            regressor = LinearRegressor()
        elif regressor_choice == "ElasticNet":
            regressor = ElasticNetRegressor()
        elif regressor_choice == "Random Forest":
            regressor = RandomForestRegression()
        elif regressor_choice == "Neural Network":
            regressor = NNRegressor()
        elif regressor_choice == "LightGBM":
            regressor = LightGBMRegressor()

        target_column = st.selectbox("Select target column for regression", st.session_state.analyzer.df.columns)
        if regressor and st.button("Train Regressor"):
            X = st.session_state.analyzer.df.drop(target_column, axis=1)
            y = st.session_state.analyzer.df[target_column]
            regressor.fit(X.values, y.values)
            st.write("Regressor trained.")
            st.write("Performance metrics:")
            st.write(regressor.get_performance_metrics(["r_squared", "mean_squared_error"]))

        # Clustering options
        st.write("## Clustering")
        clustering_options = ["KMeans", "Agglomerative", "DBSCAN", "Gaussian Mixture"]
        clustering_choice = st.selectbox("Choose a clustering algorithm", clustering_options)
        clustering = None

        if clustering_choice == "KMeans":
            clustering = KMeansClustering()
        elif clustering_choice == "Agglomerative":
            clustering = AgglomerativeClusteringModel()
        elif clustering_choice == "DBSCAN":
            clustering = DBSCANClustering()
        elif clustering_choice == "Gaussian Mixture":
            clustering = GaussianMixtureClustering()

        if clustering and st.button("Cluster Data"):
            clustering.fit(st.session_state.analyzer.df.values)
            st.write("Data clustered.")
            st.write("Cluster labels:")
            st.write(clustering.get_cluster_labels())

if __name__ == "__main__":
    main()