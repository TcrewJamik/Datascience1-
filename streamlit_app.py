import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, roc_curve, auc, confusion_matrix, classification_report
import numpy as np
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Anneal DataSet", page_icon="⚙️", layout="wide")
file_path = "anneal.data"

@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path, sep=",", header=None, na_values=["?"])
    data.columns = ["famiily", "product-type", "steel", "carbon", "hardness", "temper-rolling", "condition", "formability",
                    "strength", "non-ageing", "surface-finish", "surface-quality", "enamelability", "bc", "bf", "bt", "bw/me",
                    "bl", "m", "chrom", "phos", "cbond", "marvi", "exptl", "ferro", "corr", "blue/bright/varn/clean",
                    "lustre", "jurofm", "s", "p", "shape", "thick", "width", "len", "oil", "bore", "packing", "class"]
    return data

data_original = load_data(file_path)
data = data_original.copy()
cols_to_drop = ["famiily", "temper-rolling", "non-ageing", "surface-finish", "enamelability", "bc", "bf", "bt", "bl", "m",
                "chrom", "phos", "cbond", "marvi", "exptl", "ferro", "corr", "blue/bright/varn/clean", "lustre", "jurofm",
                "s", "p", "oil", "packing", "bw/me"]
data.drop(columns=cols_to_drop, inplace=True)
data.drop(columns=['carbon', 'hardness', 'strength', 'bore', 'product-type'], inplace=True)
data.dropna(subset=["class"], inplace=True)
class_counts = data["class"].value_counts()
if len(data["class"].unique()) > 2:
    median_freq = class_counts.median()
    group1 = class_counts[class_counts >= median_freq].index.tolist()
    data["binary_class"] = data["class"].apply(lambda x: 1 if x in group1 else 0)
else:
    data["binary_class"] = data["class"]
data.drop('class', axis=1, inplace=True)
categorical_cols = data.select_dtypes(include=['object']).columns
orig_categorical_cols = list(categorical_cols)
for col in categorical_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)
data['formability'].fillna(data['formability'].median(), inplace=True)
if 'label_encoders' not in st.session_state:
    st.session_state['label_encoders'] = {}
for col in orig_categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    st.session_state['label_encoders'][col] = le
X = data.drop('binary_class', axis=1)
y = data['binary_class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
global_scaler = StandardScaler()
X_train_scaled = pd.DataFrame(global_scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(global_scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
st.title("⚙️ Anneal DataSet")
with st.sidebar:
    st.header("🛠️ Настройки модели")
    model_choice = st.selectbox("Выберите модель:", ["KNN", "Logistic Regression", "Decision Tree"])
    hyperparams = {}
    if model_choice == "KNN":
        hyperparams['n_neighbors'] = st.slider("n_neighbors", 1, 20, 3, 1)
        hyperparams['weights'] = st.selectbox("weights", ['uniform', 'distance'], 0)
        hyperparams['metric'] = st.selectbox("metric", ['minkowski', 'euclidean', 'manhattan', 'chebyshev'], 0)
        hyperparams['p'] = st.slider("p", 1, 5, 2, 1)
    elif model_choice == "Logistic Regression":
        hyperparams['C'] = st.slider("C", 0.001, 10.0, 1.0, 0.01, format="%.3f")
        penalty_options = ['l1', 'l2', 'none']
        hyperparams['penalty'] = st.selectbox("penalty", penalty_options, 1)
        solver_options = ['lbfgs', 'liblinear']
        if hyperparams['penalty'] == 'l1':
            solver_options = ['liblinear', 'saga']
        elif hyperparams['penalty'] == 'none':
            solver_options = ['lbfgs', 'newton-cg', 'sag', 'saga']
        hyperparams['solver'] = st.selectbox("solver", solver_options, 0)
    elif model_choice == "Decision Tree":
        hyperparams['criterion'] = st.selectbox("criterion", ['gini', 'entropy'], 0)
        hyperparams['max_depth'] = st.slider("max_depth", 1, 20, 5, 1)
        hyperparams['min_samples_split'] = st.slider("min_samples_split", 2, 20, 2, 1)
        hyperparams['min_samples_leaf'] = st.slider("min_samples_leaf", 1, 10, 1, 1)
        hyperparams['max_features'] = st.selectbox("max_features", ['auto', 'sqrt', 'log2', None], 3)
    st.markdown("---")
    st.header("📊 Выбор признаков")
    available_features = X_train_scaled.columns.tolist()
    default_features = ['formability', 'condition'] if all(f in available_features for f in ['formability', 'condition', 'surface-quality', 'shape', 'steel', 'thick', 'width', 'len']) else available_features[:min(2, len(available_features))]
    selected_features = st.multiselect("Выберите признаки для обучения:", available_features, default=default_features)
    st.markdown("---")
    st.header("Настройка значений для предсказания")
    prediction_data = {}
    if selected_features:
        for feature in selected_features:
            if feature in orig_categorical_cols:
                allowed = list(map(str, st.session_state['label_encoders'][feature].classes_))
                prediction_data[feature] = st.selectbox(feature, options=allowed, index=0)
            else:
                min_val = float(X_train_scaled[feature].min())
                max_val = float(X_train_scaled[feature].max())
                default_val = float(X_train_scaled[feature].mean())
                prediction_data[feature] = st.slider(feature, min_val, max_val, default_val)
    run_button = st.button("Предсказать")
if run_button or not st.session_state.get('models_trained', False):
    st.session_state['models_trained'] = True
    if len(selected_features) < 2:
        st.warning("Выберите как минимум два признака для обучения моделей.")
        X_train_sel = X_train_scaled[default_features[:min(2, len(default_features))]]
        X_test_sel = X_test_scaled[default_features[:min(2, len(default_features))]]
    else:
        X_train_sel = X_train_scaled[selected_features]
        X_test_sel = X_test_scaled[selected_features]
    num_sel = [f for f in selected_features if f not in orig_categorical_cols]
    if num_sel:
        scaler_sel = StandardScaler()
        X_train_sel[num_sel] = scaler_sel.fit_transform(X_train_sel[num_sel])
        X_test_sel[num_sel] = scaler_sel.transform(X_test_sel[num_sel])
        st.session_state['scaler_sel'] = scaler_sel
    if model_choice == "KNN":
        classifier = KNeighborsClassifier(**hyperparams)
    elif model_choice == "Logistic Regression":
        classifier = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', **hyperparams)
    elif model_choice == "Decision Tree":
        classifier = DecisionTreeClassifier(random_state=42, **hyperparams)
    else:
        classifier = LogisticRegression()
    classifier.fit(X_train_sel, y_train)
    y_pred = classifier.predict(X_test_sel)
    y_prob = classifier.predict_proba(X_test_sel)[:, 1]
    st.session_state['classifier'] = classifier
    st.session_state['X_train_sel'] = X_train_sel
    st.session_state['X_test_sel'] = X_test_sel
    st.session_state['y_test'] = y_test
    st.session_state['y_pred'] = y_pred
    st.session_state['y_prob'] = y_prob
    st.session_state['model_choice'] = model_choice
    st.session_state['hyperparams'] = hyperparams
    st.session_state['selected_features'] = selected_features
st.header("Оценка модели")
if st.session_state.get('models_trained', False):
    st.subheader(f"Модель: {st.session_state['model_choice']}")
    st.write(f"Гиперпараметры: {st.session_state['hyperparams']}")
    col_metrics, col_charts = st.columns(2)
    with col_metrics:
        st.metric("Accuracy", f"{accuracy_score(st.session_state['y_test'], st.session_state['y_pred']):.3f}")
        st.metric("ROC AUC", f"{roc_auc_score(st.session_state['y_test'], st.session_state['y_prob']):.3f}")
        st.metric("F1", f"{f1_score(st.session_state['y_test'], st.session_state['y_pred']):.3f}")
        st.metric("Precision", f"{precision_score(st.session_state['y_test'], st.session_state['y_pred']):.3f}")
        st.metric("Recall", f"{recall_score(st.session_state['y_test'], st.session_state['y_pred']):.3f}")
    with col_charts:
        cm = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred'])
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', ax=ax_cm)
        ax_cm.set_xlabel('Предсказанные')
        ax_cm.set_ylabel('Истинные')
        ax_cm.set_title('Матрица ошибок')
        st.pyplot(fig_cm)
        fpr, tpr, _ = roc_curve(st.session_state['y_test'], st.session_state['y_prob'])
        roc_auc_val = auc(fpr, tpr)
        fig_roc = px.area(x=fpr, y=tpr, title=f'ROC (AUC = {roc_auc_val:.2f})',
                          labels=dict(x='FPR', y='TPR'))
        fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        fig_roc.update_traces(fillcolor='rgba(99, 255, 132, 0.6)')
        st.plotly_chart(fig_roc)
    st.subheader("Отчет о классификации")
    st.text(classification_report(st.session_state['y_test'], st.session_state['y_pred']))
    st.header("Результаты на тестовом наборе")
    results_df = st.session_state['X_test_sel'].copy()
    results_df['Истинный класс'] = st.session_state['y_test'].values
    results_df['Предсказанный класс'] = st.session_state['y_pred']
    results_df['Вероятность класса 1'] = st.session_state['y_prob']
    st.dataframe(results_df)
if run_button and st.session_state.get('models_trained', False) and prediction_data:
    single_prediction_df = pd.DataFrame([prediction_data])
    single_prediction_df = single_prediction_df[st.session_state['selected_features']].copy()
    for col in st.session_state['selected_features']:
        if col in orig_categorical_cols:
            single_prediction_df[col] = single_prediction_df[col].astype(str)
            le = st.session_state['label_encoders'][col]
            for val in single_prediction_df[col]:
                if val not in le.classes_:
                    st.warning(f"Категория '{val}' для '{col}' не была видна. Выберите из {', '.join(le.classes_)}")
                    st.stop()
            single_prediction_df[col] = le.transform(single_prediction_df[col])
    num_sel = [f for f in st.session_state['selected_features'] if f not in orig_categorical_cols]
    if num_sel:
        single_prediction_df[num_sel] = st.session_state['scaler_sel'].transform(single_prediction_df[num_sel])
    single_prediction = st.session_state['classifier'].predict(single_prediction_df)
    single_prediction_proba = st.session_state['classifier'].predict_proba(single_prediction_df)[:, 1]
    st.subheader("Результат предсказания")
    st.write(f"Класс: {single_prediction[0]}")
    st.write(f"Вероятность класса 1: {single_prediction_proba[0]:.3f}")
st.markdown("---")
st.markdown("Разработано компанией Jamshed Corporation совместно с ZyplAI")
