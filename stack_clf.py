#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Import Libraries
import statistics
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt


# Import xgboost
import xgboost as xgb

# Import sci-kit learn modules
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
#from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
#from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neural_network import MLPClassifier
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif


# import mlxtend for stacking
from mlxtend.classifier import StackingCVClassifier

# import bayes_opt for super learner
#from bayes_opt import BayesianOptimization

seed=123

# Read in datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")


# Remove Id columns but store test ids
train = train_df.drop(['Id'], axis = 1)
test_ids = test_df['Id']
test = test_df.drop(['Id'], axis = 1)

# Add new features (taken from kaggle/...)

def add_features(X_):
    X = X_.copy()
    
    X['Hydro_Elevation_diff'] = X[['Elevation',
                                   'Vertical_Distance_To_Hydrology']
                                  ].diff(axis='columns').iloc[:, [1]]

    X['Hydro_Euclidean'] = np.sqrt(X['Horizontal_Distance_To_Hydrology']**2 +
                                   X['Vertical_Distance_To_Hydrology']**2)

    X['Hydro_Fire_sum'] = X[['Horizontal_Distance_To_Hydrology',
                             'Horizontal_Distance_To_Fire_Points']
                            ].sum(axis='columns')

    X['Hydro_Fire_diff'] = X[['Horizontal_Distance_To_Hydrology',
                              'Horizontal_Distance_To_Fire_Points']
                             ].diff(axis='columns').iloc[:, [1]].abs()

    X['Hydro_Road_sum'] = X[['Horizontal_Distance_To_Hydrology',
                             'Horizontal_Distance_To_Roadways']
                            ].sum(axis='columns')

    X['Hydro_Road_diff'] = X[['Horizontal_Distance_To_Hydrology',
                              'Horizontal_Distance_To_Roadways']
                             ].diff(axis='columns').iloc[:, [1]].abs()

    X['Road_Fire_sum'] = X[['Horizontal_Distance_To_Roadways',
                            'Horizontal_Distance_To_Fire_Points']
                           ].sum(axis='columns')

    X['Road_Fire_diff'] = X[['Horizontal_Distance_To_Roadways',
                             'Horizontal_Distance_To_Fire_Points']
                            ].diff(axis='columns').iloc[:, [1]].abs()
    
    # Compute Soil_Type number from Soil_Type binary columns
    X['Stoneyness'] = sum(i * X['Soil_Type{}'.format(i)] for i in range(1, 41))
    
    # For all 40 Soil_Types, 1=rubbly, 2=stony, 3=very stony, 4=extremely stony, 0=?
    stoneyness = [4, 3, 1, 1, 1, 2, 0, 0, 3, 1, 
                  1, 2, 1, 0, 0, 0, 0, 3, 0, 0, 
                  0, 4, 0, 4, 4, 3, 4, 4, 4, 4, 
                  4, 4, 4, 4, 1, 4, 4, 4, 4, 4]
    
    # Replace Soil_Type number with "stoneyness" value
    X['Stoneyness'] = X['Stoneyness'].replace(range(1, 41), stoneyness)
    
    return X


# Add new features to both train and test sets
new_train = add_features(train)
new_test = add_features(test)


# Concatenate train/test for looping through to create new Soil_Type feature
data = pd.concat((new_train, new_test))


# Get 'Soil_Type' start and end index
soil_start_idx = data.columns.get_loc("Soil_Type1") 
soil_end_idx = data.columns.get_loc("Soil_Type9") 


# Create new labels for soil type
label = 1
for idx in range(soil_start_idx, soil_end_idx+1):
    data.iloc[:, idx] = data.iloc[:, idx].map({1:label, 0:0})
    label += 1

    
# Add columns together into one single column: 'Soil_Type'
data['Soil_Type'] = pd.Series([])
for idx in range(soil_start_idx, soil_end_idx+1):
    data['Soil_Type'] = data['Soil_Type'].add(data.iloc[:, idx], fill_value=0)
    

# Converted labels for Soil_Type
data['Soil_Type'] = data['Soil_Type'].astype('int').astype('category')



# Drop old Soil_Type dummy variables from full dataframe
idx_list = []
for idx in range(soil_start_idx, soil_end_idx+1):
    idx_list.append(idx)
    
# Assign new dataframe to: df    
df = data.drop(data.columns[[idx_list]], axis=1)

# Split new dataframe back to train/test split
df_train = df.iloc[:len(train), :]
df_test = df.iloc[len(train):, :]
df_test = df_test.drop('Cover_Type', axis=1)




## Mean target encoding functions

# Mean encoding function for the test set
def test_mean_target_encoding(train, test, target, categorical, alpha=5):
    # Calculate global mean on the train data
    global_mean = train[target].mean()
    
    # Group by the categorical feature and calculate its properties
    train_groups = train.groupby(categorical)
    category_sum = train_groups[target].sum()
    category_size = train_groups.size()
    
    # Calculate smoothed mean target statistics
    train_statistics = (category_sum + global_mean * alpha) / (category_size + alpha)
    
    # Apply statistics to the test data and fill new categories
    test_feature = test[categorical].map(train_statistics).fillna(global_mean)
    return test_feature.values

# Mean encoding function for the training set
def train_mean_target_encoding(train, target, categorical, alpha=5):
    # Create 5-fold cross-validation
    kf = KFold(n_splits=5, random_state=123, shuffle=True)
    train_feature = pd.Series(index=train.index)
    
    # For each folds split
    for train_index, test_index in kf.split(train):
        cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
      
        # Calculate out-of-fold statistics and apply to cv_test
        cv_test_feature = test_mean_target_encoding(cv_train, cv_test, target, categorical, alpha)
        
        # Save new feature for this particular fold
        train_feature.iloc[test_index] = cv_test_feature       
    return train_feature.values


# Function for creating the new train/test features
def mean_target_encoding(train, test, target, categorical, alpha=5):
  
    # Get the train feature
    train_feature = train_mean_target_encoding(train, target, categorical, alpha)
  
    # Get the test feature
    test_feature = test_mean_target_encoding(train, test, target, categorical, alpha)
    
    # Return new features to add to the model
    return train_feature, test_feature


# Replace original Soil_type feature with new mean target encoding features
df_train['Soil_Type'], df_test['Soil_Type'] = mean_target_encoding(df_train, 
                                                                   df_test, 
                                                                   target='Cover_Type', 
                                                                   categorical='Soil_Type',
                                                                   alpha=5)


# # Target Variable to be seperated from training set
target_variable = df_train.pop('Cover_Type')

# # Convert labels so they are accepted by model
le = LabelEncoder()
le.fit(target_variable)

# # Converted labels set: y
y = le.transform(target_variable)




# Create Train and Validation sets
X_train, X_val, y_train, y_val = train_test_split(df_train, y, test_size=0.2, random_state=seed)

###############################################################################

# Feature selection

# Keep 5 features
selector = SelectKBest(mutual_info_classif, k=20)

X_new = selector.fit_transform(X_train, y_train)


# Get back the features we've kept, zero out all other features
selected_features = pd.DataFrame(selector.inverse_transform(X_new), 
                                 index=X_train.index,
                                 columns=X_train.columns
                                 )
selected_features.head()

# Dropped columns have values of all 0s, so var is 0, drop them
selected_columns = selected_features.columns[selected_features.var() != 0]

# Get the valid dataset with the selected features.
X_train = X_train[selected_columns]
X_val = X_val[selected_columns]



###############################################################################


# Train on both sets
model1 = xgb.XGBClassifier(objective="multi:softmax",
                          n_estimators=500, 
                          max_depth=54,
                          learning_rate=0.1,
                          gamma=0.0,
                          colsample_bytree=0.9,
                          N_jobs=4,
                          seed=seed)


# Train on both sets
model2 = RandomForestClassifier(n_jobs=-1, 
                                max_depth=200, 
                                n_estimators=910, 
                                max_features='auto',
                                random_state=seed)

# Train on both sets
model3 = ExtraTreesClassifier(n_jobs=-1, 
                                max_depth=156, 
                                n_estimators=1000, 
                                max_features='auto', 
                                random_state=seed)


##
#model4 = KNeighborsClassifier(n_jobs=-1, 
#                               leaf_size=10, 
#                               n_neighbors=3, 
#                               p=1, 
#                               weights='distance')
#
#model5 = AdaBoostClassifier(n_estimators=1000,
#                             learning_rate=0.01)


#
#mlp = MLPClassifier(early_stopping=True,
#                    validation_fraction=0.2,
#                    n_iter_no_change=20,
#                    activation='relu',
#                    alpha=0.001,
#                    learning_rate_init=0.01, #best is 0.01
#                    solver='adam',
#                    verbose=True,
#                    max_iter=500,
#                    random_state=123)

###############################################################################


mlp = MLPClassifier(hidden_layer_sizes=(6000),
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                    activation='relu',
                    alpha=0.001,
                    learning_rate_init=0.01, #best is 0.01
                    solver='adam',
                    verbose=True,
                    max_iter=500,
                    random_state=seed)

###############################################################################



sclf = StackingCVClassifier(classifiers=[model1, model2, model3],
                            use_probas=True,
                            meta_classifier=mlp,
                            cv=7,
                            store_train_meta_features=True,
                            stratify=True,
                            verbose=3,
                            n_jobs=-1,
                            random_state=seed)

sclf_cv_score = cross_val_score(sclf, df_train[selected_columns].values, y=y, scoring='accuracy', cv=3)
print(f"Mean accuracy {sclf_cv_score.mean(): .4f}")
print(f"+/- {sclf_cv_score.std(): .2f}")






###############################################################################


predictions = sclf.predict(df_test[selected_columns].values) # Add values attribute to rid of 'feature_names mismatch'
final_pred= le.inverse_transform(predictions)
final_pred = [int(i) for i in final_pred]


print(final_pred)


# Model voting submission
output = pd.DataFrame({'Id':test_ids, 'Cover_Type':final_pred})
output.to_csv('submission48.csv', index=False, header=True)







