import numpy as np
import pandas as pd
import math
import pickle
from io import StringIO, BytesIO

# sklearn packages
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score, precision_recall_curve, f1_score, 
                            fbeta_score, confusion_matrix, roc_auc_score, roc_curve, make_scorer)
from sklearn.model_selection import (learning_curve, cross_val_score, cross_validate, KFold, StratifiedKFold,
                                    cross_val_predict, train_test_split, GridSearchCV)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, BaggingRegressor) 

from sklearn.svm import SVC

# statsmodels import
from statsmodels.tools import add_constant
import patsy

# import plotting tools
import matplotlib.pyplot as plt
import seaborn as sns


def replace_values(df, col_name, dict_name):
    '''Function replaces current standardized values with their categorical counterparts per the  
    attribute information found on the source data site. 
    
    Parameters:
        df: DataFrame object
        col_name: column name whose values are to be replaced
        dict_name: name of dictionary that contains the new values that will be substituted in

    Returns:
        updated pandas dataframe.  
    '''

    for k, v in dict_name.items():
        df.replace({col_name:{k:v}}, value = None,inplace = True)

def bias_metric(y_true, y_pred):
    '''Calculates MSE

    Parameters:
        y_true: array of target values
        y_pred: array of predicted values
    '''
    return np.mean((np.mean(y_pred) - y_true)**2)

def variance_metric(y_true, y_pred):
    '''Calculates variance

    Parameters:
        y_true: array of target values
        y_pred: array of predicted values
    '''
    return np.var(y_true - y_pred)

def dummy_pred(X_test, user_cat):
    '''takes an array or matrix of demographics and usage history and outputs user category prediction of 'user' in a list 
    of the same length as input array/matrix. 

    Parameters:
        X_test: array of parameters and data
        user_cat: array of demographin information 
    '''
    return [user_cat] * np.array(X_test).shape[0]

def Sort_Tuple(tup):  
    '''Sorts list of tuples by second value in descending order
    
    Parameters:
        tup: list of tuples
    '''
    lst = len(tup)  
    for i in range(0, lst):  
          
        for j in range(0, lst-i-1):  
            if (tup[j][1] < tup[j + 1][1]):  
                temp = tup[j]  
                tup[j]= tup[j + 1]  
                tup[j + 1]= temp  
    return tup  

def make_confusion_matrix(model, X_test, y_test, threshold=0.35):
    '''Creates and plots confusion matrix with specified threshold. Saves figure as 'Confusion_CV.png'.
    
    Parameters:
        model: algorithm with parameters
        threshold: float used to specify division between category 1 and 0
    '''
    # Predict class 1 if probability of being in class 1 is greater than threshold
    # (model.predict(X_test) does this automatically with a threshold of 0.5)
    y_predict = (model.predict_proba(X_test)[:, 1] >= threshold)
    fraud_confusion = confusion_matrix(y_test, y_predict)
    plt.figure(dpi=100)
    sns.heatmap(fraud_confusion, cmap=plt.cm.Blues, annot=True, square=True, fmt='d',
           xticklabels=['non-user', 'user'],
           yticklabels=['non-user', 'user']);
    plt.xlabel('prediction')
    plt.ylabel('actual')
    plt.figure(figsize = (10, 10))
    plt.savefig('Confusion_CV.png');

def new_val_and_replace(df, col_name, replacement_list):
    '''Replaces a specific column in data frame with new data from a replacement list.
    
    Parameters:
        df: pandas dataframe
        col_name: string name of column to change
        replacement_list: list object of values to replace current values in column
    '''
    # id unique values in column
    unique_val = sorted(df[col_name].unique())
    
    # make sure unique values count == replacement list contents
    try:
        len(unique_val) == len(replacement_list)
    # if length do not match output ValueError
    except ValueError:
        print(f'Replacement list length must match {col_name} length in dataframe')
        print(f'There are {shape(unique_val)[0]} unique items in the column, but fCVreplacement list has {len(replacement_list)} items.')
    
    # create a dictionary of unique values as keys and new values as values, then use to replace df values
    val_dict = {unique_val[i]: replacement_list[i] for i in range(len(unique_val))}
    replace_values(df, col_name, val_dict)
    
    # Verify change
    print(f'New {col_name} values:\n {df[col_name].unique()}\n')
    print(f'Description:\n {df[col_name].describe()}')
    return df

def run_models(X, y):
    '''Takes in parameter data and target data, performs train_test_split then runs logistic regression, KNN, Naive Bayes, SVM, Decision Tree and Random Forest on data. 
    
    Parameters:
        X: Dataframe of parameter data
        y: Series of target data
  
    Returns:
        Dataframe of metrics (accuracy, precision, recall, f1, AUC) and plot of ROC curve for each algorithm.  
    '''
    # Instantiating models
    Logistic_Regression = LogisticRegression(solver= 'liblinear', C=1000)
    K_Nearest_Neighbors = KNeighborsClassifier()
    Naive_Bayes = GaussianNB()
    SVM = SVC(gamma = 'auto', probability = True)
    Decision_Tree = DecisionTreeClassifier()
    Random_Forest = RandomForestClassifier(n_estimators = 100)

    # setting up empty dataframe, list for models to run and dictionary for legent titles.
    scores = pd.DataFrame(index=['Accuracy','Precision','Recall', 'F1'])
    models = [Logistic_Regression, K_Nearest_Neighbors, Naive_Bayes, SVM, Decision_Tree, Random_Forest]
    scores_col = {Logistic_Regression:'Logistic Regression',
                  K_Nearest_Neighbors:'KNN',
                  Naive_Bayes:'Naive Bayes',
                  SVM:'SVM',
                  Decision_Tree:'Decision Tree',
                  Random_Forest:'Random Forest'}
    
    # Train, test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 9)
    
    # Initiate figure
    plt.figure(dpi=300)
    
    for model, col_name in scores_col.items():
        model_scores = [] # accuracy, precision, recall, f1
        model.fit(X_train, y_train)
        model_scores.append(model.score(X_test, y_test)) # accuracy
        
        y_pred = model.predict(X_test)
        
        model_scores.append(precision_score(y_test, y_pred)) # precision
        model_scores.append(recall_score(y_test, y_pred)) # recall
        model_scores.append(f1_score(y_test, y_pred)) # f1
      
        scores[scores_col[model]] = model_scores
        pd.set_option('precision', 4)
        
        # Calculating ROC
        fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
        
        # Plotting ROC
        plt.plot(fpr, tpr,lw=2,label=col_name)
        plt.plot([0,1],[0,1],c='violet',ls='--')
        plt.xlim([-0.05,1.05])
        plt.ylim([-0.05,1.05])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(f'ROC curve')
    plt.legend()
    return scores

def run_CVmodels(X, y):
    '''Takes in parameter data and target data, performs train test split then cross-validation and runs logistic regression, KNN, Naive Bayes, SVM, Decision Tree and Random Forest on data. 
    
    Parameters:
        X: Dataframe of parameter data
        y: Series of target data
  
    Returns:
        Dataframe of metrics (accuracy, precision, recall, f1, AUC) and plot of ROC curve for each algorithm.  
    '''
    # Instantiating models
    Logistic_Regression = LogisticRegression(solver= 'liblinear', C=1000)
    K_Nearest_Neighbors = KNeighborsClassifier()
    Naive_Bayes = GaussianNB()
    SVM = SVC(gamma = 'auto', probability = True)
    Decision_Tree = DecisionTreeClassifier()
    Random_Forest = RandomForestClassifier(n_estimators = 100)

    # setting up empty dataframe, list for models to run and dictionary for legent titles.
    scoresCV = pd.DataFrame(index=['Accuracy (train)','Accuracy (test)','Precision','Recall', 'F1', 'AUC']) #'Accuracy(test)',
    models = [Logistic_Regression, K_Nearest_Neighbors, Naive_Bayes, SVM, Decision_Tree, Random_Forest]
    scores_col = {Logistic_Regression:'Logistic Regression',
                  K_Nearest_Neighbors:'KNN',
                  Naive_Bayes:'Naive Bayes',
                  SVM:'SVM',
                  Decision_Tree:'Decision Tree',
                  Random_Forest:'Random Forest'}
    
    # Train, test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 9)
    
    # Initiate figure
    plt.figure(dpi=300)
    
    # Execute all models
    for model, col_name in scores_col.items():
        model_scores = [] 
        output = cross_validate(
                model, 
                X_train, 
                y_train, 
                cv= StratifiedKFold(n_splits = 5, shuffle = True, random_state = 9), 
                scoring = ['accuracy',
                           'precision',
                           'recall',
                           'f1',
                           'roc_auc'
                          ], 
                return_train_score = True, 
                return_estimator =True,
                )
        
        # append scores to list
        model_scores.append(np.mean(output['train_accuracy'])) # accuracy train
        model_scores.append(np.mean(output['test_accuracy'])) # accuracy test
        model_scores.append(np.mean(output['train_precision'])) # precision
        model_scores.append(np.mean(output['train_recall'])) # recall
        model_scores.append(np.mean(output['train_f1'])) #F1
        model_scores.append(np.mean(output['train_roc_auc'])) # AUC
        
        scoresCV[scores_col[model]] = model_scores
        pd.set_option('precision', 4)
        
    
        # calculate ROC curve
        y_pred = cross_val_predict(model, X_test, y_test, 
                                   cv=StratifiedKFold(n_splits = 5, shuffle = True, random_state = 9), 
                                   method='predict_proba')
        fpr, tpr, thresholds = roc_curve(y_test, y_pred[:,1]) #how do we get y_pred using cross_validate?!!
    
        plt.plot(fpr, tpr,lw=2,label=col_name)
        plt.plot([0,1],[0,1],c='violet',ls='--')
        plt.xlim([-0.05,1.05])
        plt.ylim([-0.05,1.05])
        
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(f'ROC curve')
    plt.legend()
    plt.savefig('ROC_CV.png')
    return scoresCV

def logis_regrCV(X, y):
    '''Takes in parameter data and target data, performs train test split then cross-validation and runs logistic regression on data. 

    Parameters:
        X: Dataframe of parameter data
        y: Series of target data

    Returns:
        Dataframe of metrics (accuracy, precision, recall, f1, AUC) and plot of ROC curve.  
    '''
    folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 9)

    # instantiate model
    model = LogisticRegression(solver= 'liblinear', C=1000)

    # set up empty dataframe
    scoresCV = pd.DataFrame(index=['Accuracy(train)','Precision','Recall', 'F1', 'AUC']) #'Accuracy(test)',

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 9)

    # initiate list to store results, which will be passed into dataframe
    model_scores = [] # accuracy, precision, recall, f1, AUC

    # calculate scores
    accuracy_train = cross_val_score(model, X_train, y_train, cv=folds, scoring='accuracy')
    precision = cross_val_score(model, X_train, y_train, cv=folds, scoring='precision')
    recall = cross_val_score(model, X_train, y_train, cv=folds, scoring='recall')
    f1 = cross_val_score(model, X_train, y_train, cv=folds, scoring='f1')
    y_pred = cross_val_predict(model, X_test, y_test, cv=folds, method='predict_proba')

    # append scores to list
    model_scores.append(np.mean(accuracy_train)) # accuracy
    model_scores.append(np.mean(precision)) # precision
    model_scores.append(np.mean(recall)) # recall
    model_scores.append(np.mean(f1)) #F1
    model_scores.append(roc_auc_score(y_test, y_pred[:,1])) # AUC

    # convert list to dataframe
    scoresCV['Logistic_RegressionCV'] = model_scores
    pd.set_option('precision', 4) # 4 is number of decimal places

    # calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred[:,1])

    # plot ROC curve
    plt.figure(dpi=300)
    plt.plot(fpr, tpr,lw=2,label='Logistic Regression')
    plt.plot([0,1],[0,1],c='violet',ls='--')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'ROC curve')
    plt.legend()

    # save ROC curve to png file
    plt.savefig('ROC_CV_logit.png')
    return scoresCV

def rand_forestCV(X, y):
    '''Takes in parameter data and target data, performs train test split then cross-validation and runs Random Forest on data.  Saves ROC as ROC_CV_rf.png 

    Parameters:
        X: Dataframe of parameter data
        y: Series of target data

    Returns:
        Dataframe of metrics (accuracy, precision, recall, f1, AUC), AUC comparison between cross_val_predict and cross_validate and plot of ROC curve.  
    '''    
    # instantiate model
    model = RandomForestClassifier(
#             n_estimators = 100, 
#             max_depth=3
                bootstrap = True,
#                 class_balance.png_weight = None,
                criterion= 'gini',
                max_depth=5,
                max_features= 'auto',
                max_leaf_nodes=None,
                min_impurity_decrease= 0.0,
                min_impurity_split=None,
                min_samples_leaf=4,
                min_samples_split= 2,
                min_weight_fraction_leaf= 0.0,
                n_estimators= 37,
                n_jobs= None,
                oob_score= True,
                random_state= 9,
                verbose= 0,
                warm_start= False
                        )

    # set up empty dataframe
    scoresCV = pd.DataFrame(
            index=['Accuracy (train)',
                'Accuracy (test)',
                'Accuracy (diff)',
                'Precision',
                'Recall', 
                'F1', 
                'AUC'
                ])

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = 0.3, 
                                                        random_state = 9
                                                    )

    # initiate list to store results, which will be passed into dataframe
    model_scores = [] # accuracy, precision, recall, f1, AUC

    # fitting model to get feature importance, accuracy, precision, recall, f1 and auc
    output = cross_validate(
            model, 
            X_train, 
            y_train, 
            cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 9), 
            scoring = ['accuracy',
                    'precision',
                    'recall',
                    'f1',
                    'roc_auc'
                    ], 
            return_train_score = True, 
            return_estimator = True,
            )

    # print feature importance list 
    for idx, estimator in enumerate(output['estimator']):
        print("\nFeatures sorted by their score for estimator {}:\n".format(idx))
        feature_importances = pd.DataFrame(estimator.feature_importances_,
                                    index = X_train.columns,
                                        columns=['importance']).sort_values('importance', ascending=False)
        print(f'{feature_importances}\n\n')

    print(f'Val AUC score with cross_validate: {np.mean(output["test_roc_auc"])}')

    # append scores to list
    model_scores.append(np.mean(output['train_accuracy']))  # accuracy train
    model_scores.append(np.mean(output['test_accuracy']))   # accuracy test
    model_scores.append(np.mean(output['train_accuracy']) - np.mean(output['test_accuracy'])) # accuracy diff
    model_scores.append(np.mean(output['train_precision'])) # precision
    model_scores.append(np.mean(output['train_recall']))    # recall
    model_scores.append(np.mean(output['train_f1']))        # F1
    model_scores.append(np.mean(output['train_roc_auc']))   # AUC

    # convert list to dataframe
    scoresCV['Random_ForestCV'] = model_scores
    pd.set_option('precision', 4) # 4 is number of decimal places

    # calculate ROC curve
    y_pred = cross_val_predict(model, 
                            X_test, 
                            y_test, 
                            cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 9), 
                            method='predict_proba')
    fpr, tpr, thresholds = roc_curve(y_test, y_pred[:,1]) #how do we get y_pred using cross_validate?!!

    print("Test AUC with cross_val_predict = ", roc_auc_score(y_test, y_pred[:,1]))

    # plot ROC curve
    plt.figure(dpi=200)
    plt.plot(fpr, tpr,lw=2,label='Random Forest')
    plt.plot([0,1],[0,1],c='violet',ls='--')
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'ROC curve (Generated from hold-out)')
    plt.legend()

    # save ROC curve to png file
    plt.savefig('ROC_CV_rf.png')
    return scoresCV