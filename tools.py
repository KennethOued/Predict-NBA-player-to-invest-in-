from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

def clean_data(df):

    print("clean data...")
    df_cleaned = df.copy()
  
    df_cleaned.drop_duplicates(keep='first', inplace=True)
    count_names = df_cleaned['Name'].value_counts()

    doubles = df_cleaned[df_cleaned['Name'].isin(count_names[count_names > 1].index.tolist())]

    # retrieve from my dataset all the doubles
    df_cleaned = df_cleaned[~df_cleaned.index.isin(doubles.index)]

    df_cleaned.loc[:,'3P%'] = (df_cleaned['3P Made'] * 100) / df_cleaned['3PA']
    df_cleaned.loc[:,'FT%'] = (df_cleaned['FTM'] * 100) / df_cleaned['FTA']
    df_cleaned.loc[:,'FG%'] = (df_cleaned['FGM'] * 100) / df_cleaned['FGA']

    # Calculate the mean, while excluding NaN values
    mean_3P_percentage = df_cleaned['3P%'].mean(skipna=True)
    mean_FT_percentage = df_cleaned['FT%'].mean(skipna=True)
    mean_FG_percentage = df_cleaned['FG%'].mean(skipna=True)

    # Impute NaN values in the dataset with the calculated mean
    df_cleaned['3P%'].fillna(mean_3P_percentage, inplace=True)
    df_cleaned['FT%'].fillna(mean_FT_percentage, inplace=True)
    df_cleaned['FG%'].fillna(mean_FG_percentage, inplace=True)


    df_cleaned['3P%'] = df_cleaned['3P%'].round(2)
    df_cleaned['FT%'] = df_cleaned['FT%'].round(2)
    df_cleaned['FG%'] = df_cleaned['FG%'].round(2)

    df_cleaned['REB'] = df_cleaned['DREB'] + df_cleaned['OREB']

    target_counts = df_cleaned['TARGET_5Yrs'].value_counts()

    # Print the counts
    print("Count of Target 0:", target_counts[0])
    print("Count of Target 1:", target_counts[1])



    return df_cleaned

models = {
    'Random Forest': RandomForestClassifier(),
    'xgboost': XGBClassifier(),
    'Logistic Regression': LogisticRegression(),
     'SVC': SVC()
   # 'KNeighbors': KNeighborsClassifier()
    
}

model_hyperparameters = {
    'Random Forest': {
        'n_estimators': [100, 300, 500, 1000],
        'max_depth': [5, 10, 15, 20],
    },
    'xgboost': {
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 10],
    },
    #'KNeighbors': {
  #      'n_neighbors': [9, 11, 15, 20, 50],
  #  },
    'Logistic Regression': {
        'C': [0.1, 0.5, 1, 10],
        'penalty': ['l1','l2']
    },
    'SVC': {
        'C': [0.05, 0.1, 1, 5, 10],
        'kernel': ['linear', 'rbf']
    }
}

preprocessing_steps = [
    ('feature_selection', SelectKBest(score_func=f_classif, k=9)),
    ('feature_scaling', StandardScaler())
]
preprocessing_pipeline = Pipeline(preprocessing_steps)
