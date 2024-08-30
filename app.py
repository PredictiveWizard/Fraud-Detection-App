import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor

# Set page configuration
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# Title and Introduction
st.title('Credit Card Fraud Detection')
st.subheader('Upload Your Transaction Data')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)
    
    # Display data preview
    st.write("Data Preview:")
    st.write(df.head())
    
    # Feature Scaling
    scaler = StandardScaler()
    df['Time_scaled'] = scaler.fit_transform(df[['Time']])
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    
    # Dimensionality Reduction with PCA
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(df[['Time_scaled', 'Amount_scaled']])
    df_pca = pd.DataFrame(data=pca_features, columns=['PCA1', 'PCA2'])
    df_pca['Class'] = df['Class']
    
    # Visualization of PCA-transformed data
    st.header('PCA Visualization')
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df_pca['PCA1'], df_pca['PCA2'], c=df_pca['Class'], cmap='coolwarm', alpha=0.6)
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    plt.title('PCA of Transactions')
    st.pyplot(fig)
    
    # Anomaly Detection Models
    st.header('Anomaly Detection Results')
    model_option = st.selectbox("Select an anomaly detection model", ["Isolation Forest", "Local Outlier Factor"])

    if model_option == "Isolation Forest":
        model = IsolationForest(contamination=0.001, random_state=42)
        df['anomaly'] = model.fit_predict(df[['Time_scaled', 'Amount_scaled']])
    elif model_option == "Local Outlier Factor":
        model = LocalOutlierFactor(n_neighbors=20, contamination=0.001)
        df['anomaly'] = model.fit_predict(df[['Time_scaled', 'Amount_scaled']])
    
    # Convert -1 (anomaly) to 1 and 1 (normal) to 0 for evaluation
    df['anomaly'] = df['anomaly'].apply(lambda x: 1 if x == -1 else 0)
    
    # Evaluation metrics
    st.subheader('Model Evaluation')
    st.write("Precision, Recall, F1-Score, and ROC-AUC for the selected model:")
    st.write(classification_report(df['Class'], df['anomaly']))
    st.write("ROC-AUC Score: ", roc_auc_score(df['Class'], df['anomaly']))

    # Visualizations for anomalies
    st.header('Anomalies Visualization')
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df_pca['PCA1'], df_pca['PCA2'], c=df['anomaly'], cmap='coolwarm', alpha=0.6)
    legend1 = ax.legend(*scatter.legend_elements(), title="Anomalies")
    ax.add_artist(legend1)
    plt.title('Anomalies Detected by the Model')
    st.pyplot(fig)
    
    # Download button
    csv = df.to_csv(index=False)
    st.download_button(label="Download Anomaly Detection Results", data=csv, file_name='anomaly_detection_results.csv', mime='text/csv')
    
    st.success('File successfully processed and results generated!')
