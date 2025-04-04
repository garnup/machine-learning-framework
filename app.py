import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from machine_learning_framework.preprocessing.analyzer import Analyzer
from machine_learning_framework.model.classifier import LogisticRegressionClassifier, RandomForestClassification, SVMClassifier, NNClassifier, LightGBMClassifier
from machine_learning_framework.model.regressor import LinearRegressor, ElasticNetRegressor, RandomForestRegression, NNRegressor, LightGBMRegressor
from machine_learning_framework.model.clustering import KMeansClustering, AgglomerativeClusteringModel, DBSCANClustering, GaussianMixtureClustering

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Upload CSV", "Visualization", "Classification Analysis", "Regression Analysis", "Clustering"])

# CSV Upload Page
if page == "Upload CSV":
    st.title("Machine Learning Framework App")
    st.write("A generic machine-learning framework for convenient data analysis.")
    st.write("Upload a CSV file to get started. Example format:")
    st.write(pd.DataFrame({'Column1': [1, 2], 'Column2': ['A', 'B']}))

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
        st.write("## Data Preprocessing")
        with st.expander("Describe Data"):
            if st.button("Describe Data"):
                st.write(st.session_state.analyzer.describe())

        with st.expander("Drop Missing Data"):
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

        # Optionally shuffle the data
        st.write("## Shuffle Data")
        if st.button("Shuffle Data"):
            analyzer = st.session_state.analyzer
            analyzer.shuffle()
            st.write("Data shuffled.")
            st.write(analyzer.df)
            st.session_state.analyzer = analyzer


# Data Visualization Page, Data Plotting
elif page == "Visualization":
    st.title("Simple Data Visualization 📊")
    analyzer = st.session_state.analyzer

    # Generate Sample Data
    #data = {"Category": ["A", "B", "C", "D"], "Values": [10, 30, 50, 20]}
    #df = pd.DataFrame(data)

    # Plot Bar Chart
    #fig, ax = plt.subplots()
    #ax.bar(df["Category"], df["Values"], color=["blue", "green", "red", "purple"])
    #st.pyplot(fig)

    # Plot Correlation Matrix as Heatmap
    if st.button("Correlation Heat Map"):
        st.write("## Correlation Heat Map")
        st.pyplot(sns.heatmap(analyzer.df.corr(), annot=True).get_figure())

    # Plot pair plots
    if st.button("Pair Plots"):
        st.write("## Pair Plots")
        pair_plots = sns.pairplot(analyzer.df)
        st.pyplot(pair_plots.fig)

    # Plot Histogram of selected column(s)
    target_columns = st.multiselect("Select histogram columns", analyzer.df.columns)
    if st.button("Histogram"):
        hist = sns.histplot(analyzer.df[target_columns])
        st.pyplot(hist.get_figure())

    # Plot Box Plot of selected column(s)
    target_columns = st.multiselect("Select boxplot columns", analyzer.df.columns)
    if st.button("Box Plot"):
        boxplot = sns.boxplot(analyzer.df[target_columns])
        st.pyplot(boxplot.get_figure())

    # Scatter plot of selected columns
    x_column = st.selectbox("Select scatterplot x column", analyzer.df.columns)
    y_column = st.selectbox("Select scatterplot y column", analyzer.df.columns)
    hue_options = ["None"] + list(analyzer.df.columns)
    hue_column = st.selectbox("Select hue column", hue_options)
    if st.button("Scatter Plot"):
        if hue_column == "None":
            scatterplot = sns.scatterplot(x=analyzer.df[x_column], y=analyzer.df[y_column])
        else:
            scatterplot = sns.scatterplot(x=analyzer.df[x_column], y=analyzer.df[y_column], hue=analyzer.df[hue_column])
        st.pyplot(scatterplot.get_figure())


elif page == "Classification Analysis":
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

    st.write("Model Parameters")
    st.write(classifier.model_params)
    updated_params = st.text_input("Update Parameters (Optional)", value="")
    if updated_params and updated_params != "":
        classifier.update_model_params(ast.literal_eval(updated_params))

    target_column = st.selectbox("Select target column for classification", st.session_state.analyzer.df.columns)
    split = st.number_input(label="Train Percentage", min_value=0.0, max_value=1.0, value=0.75)
    random_state = st.number_input(label="Random State", value=1)
    train_df, test_df = classifier.get_train_test_data(st.session_state.analyzer.df, split, random_state)

    if classifier and st.button("Train Classifier"):
        X = train_df.drop(target_column, axis=1)
        y = train_df[target_column]
        classifier.fit(X.values, y.values)
        st.write("Classifier trained. Training results:")
        st.dataframe(pd.DataFrame(classifier.get_classification_report()).transpose())
        st.write("Test Results:")
        y_pred = classifier.predict(test_df.drop(target_column, axis=1))
        y_test = test_df[target_column]
        from sklearn.metrics import classification_report
        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

    param_grid = st.text_input("Parameter Grid (Optional)", value="")
    if classifier and st.button("Optimize Hyperparameters"):
        X = st.session_state.analyzer.df.drop(target_column, axis=1)
        y = st.session_state.analyzer.df[target_column]
        gs, best_params = classifier.optimize_hyperparameters(X, y, ast.literal_eval(param_grid))
        st.write("Best params:")
        st.write(best_params)


elif page == "Regression Analysis":
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

    st.write("Model Parameters")
    st.write(regressor.model_params)
    updated_params = st.text_input("Update Parameters (Optional)", value="")
    if updated_params and updated_params != "":
        regressor.update_model_params(ast.literal_eval(updated_params))

    target_column = st.selectbox("Select target column for regression", st.session_state.analyzer.df.columns)
    split = st.number_input(label="Train Percentage", min_value=0.0, max_value=1.0, value=0.75)
    random_state = st.number_input(label="Random State", value=1)
    train_df, test_df = regressor.get_train_test_data(st.session_state.analyzer.df, split, random_state)
    if regressor and st.button("Train Regressor"):
        X = train_df.drop(target_column, axis=1)
        y = train_df[target_column]
        regressor.fit(X.values, y.values)
        y_pred = regressor.predict(test_df.drop(target_column, axis=1))
        st.write("Regressor trained.")
        st.write("Training performance metrics:")
        st.write(regressor.get_performance_metrics(["r_squared", "mean_squared_error"]))
        st.write("Test performance metrics")
        from sklearn.metrics import r2_score
        from sklearn.metrics import mean_squared_error
        test_results = { "r_squared": r2_score(test_df[target_column], y_pred), "mean_squared_error": mean_squared_error(test_df[target_column], y_pred) }
        st.write(test_results)

    param_grid = st.text_input("Parameter Grid (Optional)", value="")
    if regressor and st.button("Optimize Hyperparameters"):
        X = st.session_state.analyzer.df.drop(target_column, axis=1)
        y = st.session_state.analyzer.df[target_column]
        gs, best_params = regressor.optimize_hyperparameters(X, y, ast.literal_eval(param_grid))
        st.write("Best params:")
        st.write(best_params)

elif page == "Clustering":
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

    st.write("Model Parameters")
    st.write(clustering.model_params)
    updated_params = st.text_input("Update Parameters (Optional)", value="")
    if updated_params and updated_params != "":
        clustering.update_model_params(ast.literal_eval(updated_params))

    if clustering and st.button("Cluster Data"):
        clustering.fit(st.session_state.analyzer.df.values)
        st.write("Data clustered.")
        st.write("Silhouette score:")
        st.write(clustering.get_silhouette_score())
        st.write("Cluster labels:")
        st.write(clustering.get_cluster_labels())

    param_grid = st.text_input("Parameter Grid (Optional)", value="")
    if clustering and st.button("Optimize Hyperparameters"):
        gs, best_params = clustering.optimize_hyperparameters(st.session_state.analyzer.df.values, ast.literal_eval(param_grid))
        st.write("Best params:")
        st.write(best_params)