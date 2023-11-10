import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,precision_score, f1_score
from sklearn.pipeline import Pipeline

from joblib import dump

from tools import clean_data
from tools import model_hyperparameters,models, preprocessing_steps, preprocessing_pipeline

import warnings
warnings.filterwarnings("ignore")



def score_classifier(X_test,y_test, pipeline):
    """
    make predictions using a pipeline that contains 
    preprocessing steps and a ML model
    """
    predicted = pipeline.predict(X_test)

    cm = confusion_matrix(y_test, predicted)

    precision = precision_score(y_test, predicted)
    f1 = f1_score(y_test, predicted)
    print("Results on test set")
    print("Confusion Matrix:")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    print("Precision test set:", precision)
    print("F1 Score test set:", f1)
    


    return predicted



def train(X_train, y_train, model_name):
    """Train your model"""

    print("Trained "+ str(model_name)+ " model")
    model = models[model_name]
    param_grid = model_hyperparameters[model_name]
    grid_search = GridSearchCV(model, param_grid, cv=7, scoring='precision')
    X_train_preprocessed = preprocessing_pipeline.fit_transform(X_train, y_train)
    grid_search.fit(X_train_preprocessed, y_train)

    results = grid_search.cv_results_
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    return best_model, best_score, results

def find_best_model(models, X_train, y_train):
    """Fine-tuning and find the best model"""

    list_models_performances = list()
    for model_name in models.keys():
        model, performance, results = train(X_train, y_train, model_name)
        list_models_performances.append((model, performance, results))
    best_model_performance = max(list_models_performances, key=lambda x: x[1])
    best_model = best_model_performance[0]
    score = best_model_performance[1]
    results_grid = best_model_performance[2]

    param_combinations = results_grid['params']
    scores = results_grid['mean_test_score']
    std_scores = results_grid['std_test_score']

    labels = ["param_{}".format(i) for i in range(len(param_combinations))]
    mean_precision = scores

    # plot validation scores in function of params
    plt.plot(labels, mean_precision, marker='o', linestyle='-')
    plt.xticks(rotation=90)
    plt.fill_between(labels, mean_precision - std_scores, mean_precision + std_scores, alpha=0.15, color='green')
    plt.xlabel("params")
    plt.ylabel("validation scores")
    plt.title("Validation scores for each param")

    print("best model: ",best_model, "validation_precision:", score)

    return best_model


# Load dataset
df = pd.read_csv("nba_logreg.csv")

# filter my data using filter_data
df_cleaned = clean_data(df)

labels = df_cleaned['TARGET_5Yrs'].values # labels
paramset = df_cleaned.drop(['TARGET_5Yrs','Name'],axis=1).columns.values
df_vals = df_cleaned.drop(['TARGET_5Yrs','Name'],axis=1).values

X_train, X_test, y_train, y_test = train_test_split(df_vals, labels, test_size=0.2, random_state=42, stratify=labels)

# get the best model
best_model = find_best_model(models, X_train, y_train)

# full pipeline taking preprocessing steps and best model
full_pipeline = Pipeline(preprocessing_steps + [('model', best_model)])

full_pipeline = full_pipeline.fit(X_train, y_train)

# See features selection importance
selected_indices = full_pipeline.named_steps['feature_selection'].get_support(indices=True)
# Get the scores for the selected features
feature_scores = full_pipeline.named_steps['feature_selection'].scores_
# Get the names features

selected_feature_names = [paramset[i] for i in selected_indices]
feature_scores_selected = [feature_scores[i] for i in selected_indices]
# Plot the importance of selected features
plt.figure(figsize=(8, 6))
plt.bar(selected_feature_names, feature_scores_selected)
plt.title("Importance of Selected Features")
plt.xlabel("Features")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.show()

dump(full_pipeline, 'full_pipeline.joblib')

#example of scoring for my test set
score_classifier(X_test, y_test, full_pipeline)

