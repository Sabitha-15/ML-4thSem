

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from lime.lime_tabular import LimeTabularExplainer


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def split_features_target(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return preprocessor, numeric_features, categorical_features

def encode_target_if_needed(y):
    label_encoder = None

    if y.dtype == 'object' or str(y.dtype) == 'category':
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        return y_encoded, label_encoder

    return y, label_encoder

def build_stacking_pipeline(preprocessor, final_model_name='logistic'):
    base_models = [
        ('dt', DecisionTreeClassifier(random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ]

    if final_model_name == 'logistic':
        final_estimator = LogisticRegression(max_iter=1000)
    elif final_model_name == 'rf':
        final_estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    elif final_model_name == 'dt':
        final_estimator = DecisionTreeClassifier(random_state=42)
    else:
        final_estimator = LogisticRegression(max_iter=1000)

    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=final_estimator,
        cv=5,
        passthrough=False
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', stacking_model)
    ])

    return pipeline


def train_and_predict(model, X_train, X_test, y_train):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred


def evaluate_classification(y_test, y_pred):
    results = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }
    return results


def compare_meta_models(X_train, X_test, y_train, y_test, preprocessor):
    meta_models = ['logistic', 'rf', 'dt']
    all_results = []

    for meta_model in meta_models:
        pipeline = build_stacking_pipeline(preprocessor, final_model_name=meta_model)
        trained_model, y_pred = train_and_predict(pipeline, X_train, X_test, y_train)
        scores = evaluate_classification(y_test, y_pred)

        row = {
            'Meta Model': meta_model,
            'Accuracy': scores['Accuracy'],
            'Precision': scores['Precision'],
            'Recall': scores['Recall'],
            'F1 Score': scores['F1 Score']
        }
        all_results.append(row)

    results_df = pd.DataFrame(all_results)
    return results_df


def get_transformed_training_data(trained_pipeline, X_train):
    preprocessor = trained_pipeline.named_steps['preprocessor']
    X_train_transformed = preprocessor.transform(X_train)

    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()

    feature_names = preprocessor.get_feature_names_out()
    return X_train_transformed, feature_names

def explain_with_lime(trained_pipeline, X_train, X_test, class_names=None, instance_index=0):
    preprocessor = trained_pipeline.named_steps['preprocessor']
    classifier = trained_pipeline.named_steps['classifier']

    X_train_transformed, feature_names = get_transformed_training_data(trained_pipeline, X_train)
    X_test_transformed = preprocessor.transform(X_test)

    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()

    explainer = LimeTabularExplainer(
        training_data=X_train_transformed,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )

    explanation = explainer.explain_instance(
        data_row=X_test_transformed[instance_index],
        predict_fn=classifier.predict_proba,
        num_features=10
    )

    return explanation

if __name__ == "__main__":
    file_path = "your_dataset.csv"
    target_column = "target"

    df = load_data(file_path)
    X, y = split_features_target(df, target_column)
    y, label_encoder = encode_target_if_needed(y)

    preprocessor, numeric_features, categorical_features = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    comparison_df = compare_meta_models(X_train, X_test, y_train, y_test, preprocessor)
    print("Meta Model Comparison:")
    print(comparison_df)

    best_meta_model = comparison_df.sort_values(by='F1 Score', ascending=False).iloc[0]['Meta Model']
    print("\nBest Meta Model:", best_meta_model)

    final_pipeline = build_stacking_pipeline(preprocessor, final_model_name=best_meta_model)
    final_pipeline, y_pred = train_and_predict(final_pipeline, X_train, X_test, y_train)

    final_scores = evaluate_classification(y_test, y_pred)
    print("\nFinal Evaluation:")
    for metric_name, metric_value in final_scores.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    if label_encoder is not None:
        class_names = label_encoder.classes_.tolist()
    else:
        unique_classes = np.unique(y_train)
        class_names = [str(cls) for cls in unique_classes]

    lime_explanation = explain_with_lime(
        trained_pipeline=final_pipeline,
        X_train=X_train,
        X_test=X_test,
        class_names=class_names,
        instance_index=0
    )

    print("\nLIME Explanation for Test Instance 0:")
    for feature, weight in lime_explanation.as_list():
        print(f"{feature}: {weight:.4f}")

    lime_explanation.save_to_file("lime_classification_explanation.html")
    print("\nLIME explanation saved as 'lime_classification_explanation.html'")