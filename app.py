import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, 
    f1_score, matthews_corrcoef, confusion_matrix, roc_curve, precision_recall_curve
)

st.set_page_config(page_title="Interactive ML Classifier Evaluator", layout="wide")
st.title("🎯 Machine Learning Classification Evaluator")
st.markdown("Upload any test dataset, map your target columns, adjust thresholds, and explore interactive metrics live.")

# Sidebar Configuration
st.sidebar.header("🛠️ Configuration")
model_choice = st.sidebar.selectbox("Select a Pre-trained Model", ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "SVM"])

model_files = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "SVM": "model/svm.joblib"
}

# File Uploader
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.write("### 📋 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    # INTERACTIVE COLUMN MAPPING
    st.sidebar.subheader("🔗 Column Mapping")
    all_columns = list(df.columns)
    
    # Automatically try to guess the target column name
    default_idx = all_columns.index('target') if 'target' in all_columns else 0
    target_col = st.sidebar.selectbox("Select Target Variable (Y)", all_columns, index=default_idx)
    
    feature_cols = st.sidebar.multiselect("Select Feature Variables (X)", [col for col in all_columns if col != target_col], default=[col for col in all_columns if col != target_col])
    
    if not feature_cols:
        st.warning("Please select at least one feature column to run evaluation.")
    else:
        X = df[feature_cols]
        # Force conversions to integers or standard numeric format to avoid matrix label crashes
        y_true = df[target_col]
        
        model_path = model_files[model_choice]
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            
            # Predict probabilities if supported
            has_proba = hasattr(model, "predict_proba")
            
            # INTERACTIVE THRESHOLD SLIDER
            st.sidebar.subheader("🎛️ Decision Threshold")
            if has_proba:
                threshold = st.sidebar.slider("Adjust Classification Threshold (Probability)", 0.0, 1.0, 0.5, 0.05)
                y_prob = model.predict_proba(X)[:, 1]
                y_pred = (y_prob >= threshold).astype(int)
            else:
                st.sidebar.info("Model doesn't support probabilities. Using default predictions.")
                y_pred = model.predict(X)
                y_prob = None

            # Calculate Performance Metrics
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_true, y_pred)
            
            # Handle situations where AUC score calculation breaks due to single-class target arrays
            try:
                auc = roc_auc_score(y_true, y_prob) if y_prob is not None else "N/A"
            except Exception:
                auc = "N/A (Requires both classes in target)"

            # Layout for metrics
            st.write(f"### 📊 Performance Metrics: {model_choice}")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("Precision", f"{prec:.4f}")
            col3.metric("Recall", f"{rec:.4f}")
            col4.metric("F1 Score", f"{f1:.4f}")
            col5.metric("MCC", f"{mcc:.4f}")
            col6.metric("AUC", f"{auc:.4f}" if isinstance(auc, float) else auc)

            # Interactive Plotly Visualizations split into tabs
            tab1, tab2, tab3 = st.tabs(["🧩 Confusion Matrix", "📈 Evaluation Curves", "🔍 Sample Inspector"])
            
            with tab1:
                st.write("#### Interactive Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                
                # Robustly map text matrices to visualization
                x_labels = ['Predicted Negative', 'Predicted Positive']
                y_labels = ['Actual Negative', 'Actual Positive']
                
                # Simplified robust imshow without text_auto parameters to prevent version bugs
                fig_cm = px.imshow(
                    cm, 
                    x=x_labels, 
                    y=y_labels, 
                    colorscale='Blues',
                    labels=dict(x="Predicted Class", y="Actual Class", color="Count")
                )
                
                # Force standard text values inside the blocks manually
                fig_cm.update_traces(text=cm, texttemplate="%{text}")
                fig_cm.update_layout(width=500, height=400, margin=dict(t=50, b=50, l=50, r=50))
                st.plotly_chart(fig_cm, use_container_width=True)

            with tab2:
                if y_prob is not None and len(set(y_true)) > 1:
                    c1, c2 = st.columns(2)
                    
                    with c1:
                        st.write("#### ROC Curve")
                        fpr, tpr, _ = roc_curve(y_true, y_prob)
                        auc_val = f"{auc:.4f}" if isinstance(auc, float) else "N/A"
                        fig_roc = px.line(x=fpr, y=tpr, labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'}, title=f"ROC Curve (AUC: {auc_val})")
                        fig_roc.add_shape(type="line", line=dict(dash='dash', color='red'), x0=0, x1=1, y0=0, y1=1)
                        st.plotly_chart(fig_roc, use_container_width=True)
                        
                    with c2:
                        st.write("#### Precision-Recall Curve")
                        precision, recall, _ = precision_recall_curve(y_true, y_prob)
                        fig_pr = px.line(x=recall, y=precision, labels={'x': 'Recall', 'y': 'Precision'}, title="Precision-Recall Curve")
                        st.plotly_chart(fig_pr, use_container_width=True)
                else:
                    st.info("Evaluation curves require probability predictions and both binary classes (0 and 1) to exist inside the dataset.")

            with tab3:
                st.write("#### Inspect Data Rows and Predictions")
                inspect_df = df.copy()
                inspect_df['Actual_Target'] = y_true
                inspect_df['Predicted_Class'] = y_pred
                if y_prob is not None:
                    inspect_df['Predicted_Probability'] = y_prob
                
                # Filter by prediction outcomes
                filter_choice = st.selectbox("Filter Rows By Outcome", ["All Samples", "True Positives (TP)", "True Negatives (TN)", "False Positives (FP)", "False Negatives (FN)"])
                
                if filter_choice == "True Positives (TP)":
                    filtered_df = inspect_df[(inspect_df['Actual_Target'] == 1) & (inspect_df['Predicted_Class'] == 1)]
                elif filter_choice == "True Negatives (TN)":
                    filtered_df = inspect_df[(inspect_df['Actual_Target'] == 0) & (inspect_df['Predicted_Class'] == 0)]
                elif filter_choice == "False Positives (FP)":
                    filtered_df = inspect_df[(inspect_df['Actual_Target'] == 0) & (inspect_df['Predicted_Class'] == 1)]
                elif filter_choice == "False Negatives (FN)":
                    filtered_df = inspect_df[(inspect_df['Actual_Target'] == 1) & (inspect_df['Predicted_Class'] == 0)]
                else:
                    filtered_df = inspect_df
                    
                st.dataframe(filtered_df, use_container_width=True)
        else:
            st.error(f"Model file not found at '{model_path}'. Please verify the path or run your training script first.")
else:
    st.info("💡 Awaiting file upload. Please drag and drop or upload your test dataset CSV in the sidebar to begin.")
