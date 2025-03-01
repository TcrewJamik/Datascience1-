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

# Загружаем данные

file_path = "anneal.data"

@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path, sep=",", header=None, na_values=["?"])
    data.columns = [
        "famiily", "product-type", "steel", "carbon", "hardness", "temper-rolling", "condition", "formability",
        "strength", "non-ageing", "surface-finish", "surface-quality", "enamelability", "bc", "bf", "bt", "bw/me",
        "bl", "m", "chrom", "phos", "cbond", "marvi", "exptl", "ferro", "corr", "blue/bright/varn/clean",
        "lustre", "jurofm", "s", "p", "shape", "thick", "width", "len", "oil", "bore", "packing", "class"
    ]
    return data

data_original = load_data(file_path)
data = data_original.copy()

# Предобработка данных
columns_to_drop = [
    "famiily", "temper-rolling", "non-ageing", "surface-finish", "enamelability", "bc", "bf", "bt", "bl", "m",
    "chrom", "phos", "cbond", "marvi", "exptl", "ferro", "corr", "blue/bright/varn/clean", "lustre", "jurofm",
    "s", "p", "oil", "packing", "bw/me"
]
data.drop(columns=columns_to_drop, inplace=True)
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
for col in categorical_cols:
    mode_value = data[col].mode()[0]
    data[col].fillna(mode_value, inplace=True)
median_formability = data['formability'].median()
data['formability'].fillna(median_formability, inplace=True)

label_encoder = LabelEncoder()
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

# Разделение данных
X = data.drop('binary_class', axis=1)
y = data['binary_class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
numerical_cols = X_train.columns
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

st.title("⚙️ Anneal DataSet")

# Слайдбары для настройки
with st.sidebar:
    st.header("🛠️ Настройки модели")
    model_choice = st.selectbox("Выберите модель:", ["KNN", "Logistic Regression", "Decision Tree"])

    hyperparams = {}
    if model_choice == "KNN":
        hyperparams['n_neighbors'] = st.slider("n_neighbors", min_value=1, max_value=20, value=3, step=1)
        hyperparams['weights'] = st.selectbox("weights", options=['uniform', 'distance'], index=0)
        hyperparams['metric'] = st.selectbox("metric", options=['minkowski', 'euclidean', 'manhattan', 'chebyshev'], index=0)
        hyperparams['p'] = st.slider("p (Minkowski distance power)", min_value=1, max_value=5, value=2, step=1)

    elif model_choice == "Logistic Regression":
        hyperparams['C'] = st.slider("C (Regularization)", min_value=0.001, max_value=10.0, step=0.01, value=1.0, format="%.3f")
        penalty_options = ['l1', 'l2', 'none']
        hyperparams['penalty'] = st.selectbox("penalty", options=penalty_options, index=1)

        solver_options = ['lbfgs', 'liblinear']
        if hyperparams['penalty'] == 'l1':
            solver_options = ['liblinear', 'saga']
        elif hyperparams['penalty'] == 'none':
            solver_options = ['lbfgs', 'newton-cg', 'sag', 'saga']

        hyperparams['solver'] = st.selectbox("solver", options=solver_options, index=0)


    elif model_choice == "Decision Tree":
        hyperparams['criterion'] = st.selectbox("criterion", options=['gini', 'entropy'], index=0)
        hyperparams['max_depth'] = st.slider("max_depth", min_value=1, max_value=20, value=5, step=1)
        hyperparams['min_samples_split'] = st.slider("min_samples_split", min_value=2, max_value=20, value=2, step=1)
        hyperparams['min_samples_leaf'] = st.slider("min_samples_leaf", min_value=1, max_value=10, value=1, step=1)
        hyperparams['max_features'] = st.selectbox("max_features", options=['auto', 'sqrt', 'log2', None], index=3)

    st.markdown("---")
    st.header("📊 Выбор признаков")
    available_features = X_train.columns.tolist()
    default_features = ['formability', 'condition'] if all(f in available_features for f in ['formability', 'surface-quality', 'shape', 'steel', 'thick', 'width', 'len']) else available_features[:min(2, len(available_features))]
    selected_features = st.multiselect("Выберите признаки для обучения:", available_features, default=default_features)
    retrain_button = st.button("Предсказать")

    st.markdown("---")
    st.header("Предсказание")
    prediction_data = {}
    if selected_features:
        for feature in selected_features:
            if feature in numerical_cols:
                min_val = float(X_train[feature].min())
                max_val = float(X_train[feature].max())
                default_val = float(X_train[feature].mean()) 
                prediction_data[feature] = st.sidebar.slider(
                    f"Выберите значение для {feature}:",
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val
                )
            elif feature in categorical_cols:
                unique_categories = data_original[feature].dropna().unique()
                if len(unique_categories) > 0:
                    prediction_data[feature] = st.sidebar.selectbox(
                        f"Выберите значение для {feature}:",
                        options=unique_categories,
                        index=0 
                    )
                else:
                    prediction_data[feature] = st.sidebar.text_input(f"Введите значение для {feature}:")
            else:
                prediction_data[feature] = st.sidebar.text_input(f"Введите значение для {feature}:")

        predict_single_button = st.sidebar.button("Предсказать класс")
    else:
        st.sidebar.warning("Выберите признаки для предсказания.")
        predict_single_button = False


# Иследование данных
expander_data_explore = st.expander("Исследование данных", expanded=False)
with expander_data_explore:
    st.subheader("Предварительный просмотр данных")
    st.dataframe(data_original.head())

    st.subheader("Описательная статистика")
    st.dataframe(data.describe())

    st.subheader("Распределение классов")
    binary_class_counts = data["binary_class"].value_counts()
    fig_class_dist = px.bar(binary_class_counts, x=binary_class_counts.index, y=binary_class_counts.values,
                             labels={'x': 'Класс', 'y': 'Количество'}, title="Распределение классов (Бинарная)")
    st.plotly_chart(fig_class_dist)

    st.subheader("Процент пропущенных значений (до обработки)")
    missing_percentage = data_original.isna().sum() / len(data_original) * 100
    missing_df = pd.DataFrame({'Признак': missing_percentage.index, 'Процент пропусков': missing_percentage.values})
    missing_df = missing_df[missing_df['Процент пропусков'] > 0].sort_values(by='Процент пропусков', ascending=False)

    if not missing_df.empty:
        fig_missing = px.bar(missing_df, x='Признак', y='Процент пропусков',
                                labels={'Процент пропусков': '% пропусков'},
                                title="Процент пропущенных значений в признаках")
        st.plotly_chart(fig_missing)
    else:
        st.info("Пропущенные значения отсутствуют или уже обработаны.")

    if st.checkbox("Показать гистограммы признаков"):
        st.subheader("Гистограммы признаков")
        feature_hist_cols = st.columns(3)
        for i, col in enumerate(X_train.columns):
            with feature_hist_cols[i % 3]:
                fig_hist, ax_hist = plt.subplots()
                sns.histplot(data=X_train, x=col, kde=True, ax=ax_hist)
                ax_hist.set_title(col, fontsize=10)
                st.pyplot(fig_hist, use_container_width=True)


# Обучение и оценка
if retrain_button or not st.session_state.get('models_trained', False):
    st.session_state['models_trained'] = True

    if len(selected_features) < 2:
        st.warning("Выберите как минимум два признака для обучения моделей. Модели обучены на дефолтных признаках.")
        X_train_selected = X_train[default_features[:min(2, len(default_features))]]
        X_test_selected = X_test[default_features[:min(2, len(default_features))]]
    else:
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]

    # Обучение моделей с разными гиперпараметрами
    if model_choice == "KNN":
        classifier = KNeighborsClassifier(**hyperparams)
    elif model_choice == "Logistic Regression":
        classifier = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced', **hyperparams) 
    elif model_choice == "Decision Tree":
        classifier = DecisionTreeClassifier(random_state=42, **hyperparams)
    else:
        classifier = LogisticRegression()

    classifier.fit(X_train_selected, y_train)
    y_pred = classifier.predict(X_test_selected)
    y_prob = classifier.predict_proba(X_test_selected)[:, 1]

    st.session_state['classifier'] = classifier
    st.session_state['X_train_selected'] = X_train_selected
    st.session_state['y_train'] = y_train
    st.session_state['X_test_selected'] = X_test_selected
    st.session_state['y_test'] = y_test
    st.session_state['y_pred'] = y_pred
    st.session_state['y_prob'] = y_prob
    st.session_state['model_choice'] = model_choice
    st.session_state['hyperparams'] = hyperparams
    st.session_state['selected_features'] = selected_features
    st.session_state['label_encoders'] = {} 


# Сохраняем LabelEncoder для каждого категориального столбца
if not st.session_state.get('label_encoders'): 
    label_encoders_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        le.fit(data_original[col].dropna().astype(str)) 
        label_encoders_dict[col] = le
    st.session_state['label_encoders'] = label_encoders_dict


# Отображение результатов моделей
st.header("🏆 Оценка модели")
if st.session_state.get('models_trained', False):
    st.subheader(f"Модель: {st.session_state['model_choice']}")
    st.write(f"Гиперпараметры: {st.session_state['hyperparams']}")

    col_metrics, col_charts = st.columns(2)
    with col_metrics:
        st.metric("Точность (Accuracy)", f"{accuracy_score(st.session_state['y_test'], st.session_state['y_pred']):.3f}")
        st.metric("ROC AUC", f"{roc_auc_score(st.session_state['y_test'], st.session_state['y_prob']):.3f}")
        st.metric("F1-мера (F1 Score)", f"{f1_score(st.session_state['y_test'], st.session_state['y_pred']):.3f}")
        st.metric("Точность (Precision)", f"{precision_score(st.session_state['y_test'], st.session_state['y_pred']):.3f}")
        st.metric("Полнота (Recall)", f"{recall_score(st.session_state['y_test'], st.session_state['y_pred']):.3f}")

    with col_charts:
        # Confusion Matri
        cm = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred'])
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', ax=ax_cm)
        ax_cm.set_xlabel('Предсказанные классы')
        ax_cm.set_ylabel('Истинные классы')
        ax_cm.set_title('Матрица ошибок (Confusion Matrix)')
        st.pyplot(fig_cm)

        # ROC
        fpr, tpr, thresholds = roc_curve(st.session_state['y_test'], st.session_state['y_prob'])
        roc_auc = auc(fpr, tpr)

        fig_roc = px.area(
            x=fpr, y=tpr,
            title=f'ROC-кривая (AUC = {roc_auc:.2f})',
            labels=dict(x='False Positive Rate', y='True Positive Rate'),
        )
        fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        fig_roc.update_traces(fillcolor='rgba(99, 255, 132, 0.6)')
        st.plotly_chart(fig_roc)

    st.subheader("Отчет о классификации")
    st.text(classification_report(st.session_state['y_test'], st.session_state['y_pred']))

    # Display Prediction Results for Test Set
    st.header("Результаты предсказания на тестовом наборе")
    results_df = pd.DataFrame(st.session_state['X_test_selected'].copy())
    results_df['Истинный класс'] = st.session_state['y_test'].values
    results_df['Предсказанный класс'] = st.session_state['y_pred']
    results_df['Вероятность класса 1'] = st.session_state['y_prob']
    st.dataframe(results_df)


if predict_single_button and st.session_state.get('models_trained', False) and prediction_data:
    single_prediction_df = pd.DataFrame([prediction_data])
    single_prediction_df = single_prediction_df[st.session_state['selected_features']].copy()

    # Предобработка 
    for col in single_prediction_df.columns:
        if col in categorical_cols:
            single_prediction_df[col] = single_prediction_df[col].astype(str)
            le = st.session_state['label_encoders'][col]
            for val in single_prediction_df[col]:
                if val not in list(le.classes_): 
                    st.warning(f"Категория '{val}' для признака '{col}' не была видна во время обучения. "
                               f"Выберите одно из: {', '.join(list(le.classes_))}")
                    st.stop() 
            single_prediction_df[col] = le.transform(single_prediction_df[[col]])
    numerical_cols_selected = [col for col in st.session_state['selected_features'] if col in numerical_cols]
    if numerical_cols_selected:
        single_prediction_df[numerical_cols_selected] = scaler.transform(single_prediction_df[numerical_cols_selected])


    single_prediction = st.session_state['classifier'].predict(single_prediction_df)
    single_prediction_proba = st.session_state['classifier'].predict_proba(single_prediction_df)[:, 1]

    st.subheader("Результат предсказания:")
    st.write(f"Предсказанный класс: **{single_prediction[0]}**")
    st.write(f"Вероятность класса 1: **{single_prediction_proba[0]:.3f}**")


st.markdown("---")
st.markdown("Разработано компанией Jamshed Corporation совместно с ZyplAI")
