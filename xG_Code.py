import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
import xgboost
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc, log_loss
import scipy
import random
import pickle
import joblib

### Import data
shots_sorted=pd.read_csv(r'/Users/kyledelaney/Downloads/shots_sorted.csv')


#### Clean data
def clean_data(shots_sorted)

    """
    1. clean the shots data
    2. add new necessary columns
    """
    ###add shot types
    shots_sorted['type_backhand']=np.where(shots_sorted['shotType'] == 'BACK', 1,0)
    shots_sorted['type_wrist']=np.where(shots_sorted['shotType'] == 'WRIST',1,0)
    shots_sorted['type_slap']=np.where(shots_sorted['shotType'] == 'SLAP',1,0)
    shots_sorted['type_tipin']=np.where(shots_sorted['shotType'] == 'TIP',1,0)
    shots_sorted['type_deflect']=np.where(shots_sorted['shotType'] == 'DEFL',1,0)
    shots_sorted['type_wrap']=np.where(shots_sorted['shotType'] == 'WRAP',1,0)
    shots_sorted['type_snap']=np.where(shots_sorted['shotType'] == 'SNAP',1,0)
    shots_sorted['shotType'].fillna("NA", inplace=True)
    
    #add strengths
    shots_sorted['strength_3x3']=np.where((shots_sorted['homeSkatersOnIce'] == 3) & (shots_sorted['awaySkatersOnIce'] == 3),1,0)
    shots_sorted['strength_3x4']=np.where((shots_sorted['homeSkatersOnIce'] == 3) & (shots_sorted['awaySkatersOnIce'] == 4),1,0)
    shots_sorted['strength_3x5']=np.where((shots_sorted['homeSkatersOnIce'] == 3) & (shots_sorted['awaySkatersOnIce'] == 5),1,0)
    shots_sorted['strength_3x6']=np.where((shots_sorted['homeSkatersOnIce'] == 3) & (shots_sorted['awaySkatersOnIce'] == 6),1,0)
    shots_sorted['strength_4x3']=np.where((shots_sorted['homeSkatersOnIce'] == 4) & (shots_sorted['awaySkatersOnIce'] == 3),1,0)
    shots_sorted['strength_4x4']=np.where((shots_sorted['homeSkatersOnIce'] == 4) & (shots_sorted['awaySkatersOnIce'] == 4),1,0)
    shots_sorted['strength_4x5']=np.where((shots_sorted['homeSkatersOnIce'] == 4) & (shots_sorted['awaySkatersOnIce'] == 5),1,0)
    shots_sorted['strength_4x6']=np.where((shots_sorted['homeSkatersOnIce'] == 4) & (shots_sorted['awaySkatersOnIce'] == 6),1,0)
    shots_sorted['strength_5x3']=np.where((shots_sorted['homeSkatersOnIce'] == 5) & (shots_sorted['awaySkatersOnIce'] == 3),1,0)
    shots_sorted['strength_5x4']=np.where((shots_sorted['homeSkatersOnIce'] == 5) & (shots_sorted['awaySkatersOnIce'] == 4),1,0)
    shots_sorted['strength_5x5']=np.where((shots_sorted['homeSkatersOnIce'] == 5) & (shots_sorted['awaySkatersOnIce'] == 5),1,0)
    shots_sorted['strength_5x6']=np.where((shots_sorted['homeSkatersOnIce'] == 5) & (shots_sorted['awaySkatersOnIce'] == 6),1,0)
    shots_sorted['strength_6x3']=np.where((shots_sorted['homeSkatersOnIce'] == 6) & (shots_sorted['awaySkatersOnIce'] == 3),1,0)
    shots_sorted['strength_6x4']=np.where((shots_sorted['homeSkatersOnIce'] == 6) & (shots_sorted['awaySkatersOnIce'] == 4),1,0)
    shots_sorted['strength_6x5']=np.where((shots_sorted['homeSkatersOnIce'] == 6) & (shots_sorted['awaySkatersOnIce'] == 5),1,0)

    ### add column for forward
    shots_sorted['isforward'] = np.where(shots_sorted['playerPositionThatDidEvent'].isin(["LW", "RW", "C"]), 1, 0)

    ### remove playoffs & shootouts
    shots_sorted=shots_sorted[shots_sorted.isPlayoffGame != 1]
    shots_sorted=shots_sorted[shots_sorted.period != 5]

    ### remove goalie shots
    shots_sorted = shots_sorted[shots_sorted.playerPositionThatDidEvent != "G"]
    shots_sorted['goalieNameForShot'].fillna("Empty", inplace=True)

    ### add outcomes
    shots_sorted['Outcome'] = np.where(shots_sorted['event'] == "GOAL", 2, np.where(shots_sorted['event'] == "SHOT", 1, np.where(shots_sorted['event'] == "MISS", 0, 3)))
    shots_sorted = shots_sorted[shots_sorted['Outcome'] != 3]

    # Change giveaway to takeaway for other team
    shots_sorted['lastEventTeam'] = np.where(shots_sorted['lastEventCategory'] != "GIVE", shots_sorted["lastEventTeam"],np.where(shots_sorted['lastEventTeam'] == 'HOME', 'AWAY', 'HOME'))
    shots_sorted['lastEventCategory'] = np.where(shots_sorted['lastEventCategory'] == "GIVE", "TAKE", shots_sorted['lastEventCategory'])

    ### add column for same team events
    shots_sorted['if_prev_ev_team'] = np.where(shots_sorted['team'] == shots_sorted['lastEventTeam'], 1, 0)

    # Get if last event was by event team for specified events
    shots_sorted['prev_evTeam_Fac'] = np.where((shots_sorted['if_prev_ev_team'] == 1) & (shots_sorted['lastEventCategory'] == "FAC"), 1, 0)
    shots_sorted['prev_evTeam_NonSog'] = np.where((shots_sorted['if_prev_ev_team'] == 1) & (shots_sorted['lastEventCategory'].isin(["MISS", "BLOCK"])), 1, 0)
    shots_sorted['prev_evTeam_NonShot'] = np.where((shots_sorted['if_prev_ev_team'] == 1) & (shots_sorted['lastEventCategory'].isin(["TAKE", "HIT"])), 1, 0)
    shots_sorted['prev_evTeam_Sog'] = np.where((shots_sorted['if_prev_ev_team'] == 1) & (shots_sorted['lastEventCategory'] == "SHOT"), 1, 0)

    # Get if last event was by non-event team for specified events
    shots_sorted['prev_non_evTeam_Fac'] = np.where((shots_sorted['if_prev_ev_team'] == 0) & (shots_sorted['lastEventCategory'] == "FAC"), 1, 0)
    shots_sorted['prev_non_evTeam_NonSog'] = np.where((shots_sorted['if_prev_ev_team'] == 0) & (shots_sorted['lastEventCategory'].isin(["MISS", "BLOCK"])),1, 0)
    shots_sorted['prev_non_evTeam_NonShot'] = np.where((shots_sorted['if_prev_ev_team'] == 0) & (shots_sorted['lastEventCategory'].isin(["TAKE", "HIT"])), 1,0)
    shots_sorted['prev_non_evTeam_Sog'] = np.where((shots_sorted['if_prev_ev_team'] == 0) & (shots_sorted['lastEventCategory'] == "SHOT"), 1, 0)

    ### add column for non-SOG rebounds
    shots_sorted['non_sog_rebound'] = np.where((shots_sorted['lastEventCategory'].isin(["MISS", "BLOCK"])) & (shots_sorted['timeUntilNextEvent'] <= 2.0)& (shots_sorted['team'] == shots_sorted['lastEventTeam']), 1, 0)
    shots_sorted['shotType'].fillna("NA", inplace=True)

    ### add score categories
    ### stop at +3 and -3
    shots_sorted['score_cat'] = np.where(shots_sorted['homeTeamGoals'] - shots_sorted['awayTeamGoals'] >= 3, 3, np.where(shots_sorted['homeTeamGoals'] - shots_sorted['awayTeamGoals'] <= -3, -3, shots_sorted['homeTeamGoals'] - shots_sorted['awayTeamGoals']))
    shots_sorted['score_cat'] = np.where(shots_sorted['teamCode'] == shots_sorted['isHomeTeam'], shots_sorted['score_cat'], -shots_sorted['score_cat'])
    shots_sorted['score_cat_3'] = np.where((shots_sorted['score_cat'] == 3), 1, 0)
    shots_sorted['score_cat_2'] = np.where((shots_sorted['score_cat'] == 2), 1, 0)
    shots_sorted['score_cat_1'] = np.where((shots_sorted['score_cat'] == 1), 1, 0)
    shots_sorted['score_cat_0'] = np.where((shots_sorted['score_cat'] == 0), 1, 0)
    shots_sorted['score_cat_-1'] = np.where((shots_sorted['score_cat'] == -1), 1, 0)
    shots_sorted['score_cat_-2'] = np.where((shots_sorted['score_cat'] == -2), 1, 0)
    shots_sorted['score_cat_-3'] = np.where((shots_sorted['score_cat'] == -3), 1, 0)

    ### only need these events
    shots_sorted = shots_sorted[shots_sorted.event.isin(["SHOT", "GOAL", "MISS"])]

    ### drop dupes
    shots_sorted.drop_duplicates(subset=['game_id', 'period', 'event', 'time'], inplace=True)

    shots_sorted = shots_sorted[shots_sorted["arenaAdjustedXCord"].notnull()]
    shots_sorted = shots_sorted[shots_sorted["arenaAdjustedYCord"].notnull()]
    shots_sorted = shots_sorted[shots_sorted["lastEventxCord_adjusted"].notnull()]
    shots_sorted = shots_sorted[shots_sorted["lastEventyCord_adjusted"].notnull()]

    return shots_sorted

#### Sort index
shots_sorted=shots_sorted.sort_index(axis=1)
first_column = shots_sorted.pop('shotID')
shots_sorted.insert(0, 'shotID', first_column)

#### Use this function for xReb Model
def convert_rebound(shots_sorted)

    """"
    Convert the data to use in model
    """"

    all_variables =

    ['arenaAdjustedShotDistance','arenaAdjustedXCord','lastEventxCord_adjusted','arenaAdjustedYCord','lastEventyCord_adjusted','shotAngleAdjusted',
     'awayEmptyNet', 'homeEmptyNet', 'speedFromLastEvent', 'shotAnglePlusReboundSpeed','shotRebound','distanceFromLastEvent',
     'timeSinceLastEvent','type_backhand', 'type_deflect', 'type_slap','type_snap', 'type_tipin', 'type_wrap', 'type_wrist',
     'strength_3x3','strength_3x4', 'strength_3x5', 'strength_3x6', 'strength_4x3','strength_4x4', 'strength_4x5', 'strength_4x6',
     'strength_5x3','strength_5x4', 'strength_5x5', 'strength_5x6', 'strength_6x3', 'strength_6x4', 'strength_6x5','score_cat_-3', 
     'score_cat_-2', 'score_cat_-1', 'score_cat_0', 'score_cat_1', 'score_cat_2', 'score_cat_3','isforward','isHomeTeam',
     'prev_evTeam_Fac', 'prev_evTeam_NonSog', 'prev_evTeam_NonShot', 'prev_evTeam_Sog','prev_non_evTeam_Fac', 
     'prev_non_evTeam_NonSog', 'prev_non_evTeam_NonShot', 'prev_non_evTeam_Sog']

    categorical_variables = ['shotType', 'score_cat', 'lastEventCategory']
    labels = ['shotGeneratedRebound']

    df_dummies = pd.get_dummies(data, columns=categorical_variables)
    model_df = df_dummies[all_variables + ["shotGeneratedRebound"]]
    model_df.dropna(inplace=True)

    model_features = model_df[all_variables].values.tolist()
    model_labels = model_df[labels].values.tolist()
    
    return model_features, model_labels


#### Use this function for xG Model
def convert_goals(shots_sorted)

    """"
    Convert the data to use in model
    """"

    all_variables =

    ['arenaAdjustedShotDistance','arenaAdjustedXCord','lastEventxCord_adjusted','arenaAdjustedYCord','lastEventyCord_adjusted','shotAngleAdjusted',
     'awayEmptyNet', 'homeEmptyNet', 'speedFromLastEvent', 'shotAnglePlusReboundSpeed','shotRebound','distanceFromLastEvent',
     'timeSinceLastEvent','type_backhand', 'type_deflect', 'type_slap','type_snap', 'type_tipin', 'type_wrap', 'type_wrist',
     'strength_3x3','strength_3x4', 'strength_3x5', 'strength_3x6', 'strength_4x3','strength_4x4', 'strength_4x5', 'strength_4x6',
     'strength_5x3','strength_5x4', 'strength_5x5', 'strength_5x6', 'strength_6x3', 'strength_6x4', 'strength_6x5','score_cat_-3', 
     'score_cat_-2', 'score_cat_-1', 'score_cat_0', 'score_cat_1', 'score_cat_2', 'score_cat_3','isforward','isHomeTeam',
     'prev_evTeam_Fac', 'prev_evTeam_NonSog', 'prev_evTeam_NonShot', 'prev_evTeam_Sog','prev_non_evTeam_Fac', 
     'prev_non_evTeam_NonSog', 'prev_non_evTeam_NonShot', 'prev_non_evTeam_Sog']

    categorical_variables = ['shotType', 'score_cat', 'lastEventCategory']
    labels = ['Outcome']

    df_dummies = pd.get_dummies(data, columns=categorical_variables)
    model_df = df_dummies[all_variables + ["Outcome"]]
    model_df.dropna(inplace=True)

    model_features = model_df[all_variables].values.tolist()
    model_labels = model_df[labels].values.tolist()
    
    return model_features, model_labels





def get_roc(actual, predictions):
    """
    Get the roc curve (and auc score) for the different models
    """
    fig = plt.figure()
    plt.title('ROC Curves')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    colors = ['b', 'g', 'p']

    for model, color in zip(predictions.keys(), colors):
        # Convert preds to just prob of goal
        preds = [pred[1] for pred in predictions[model]]

        false_positive_rate, true_positive_rate, thresholds = roc_curve(flat_test_labels, preds, pos_label=1)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        plt.plot(false_positive_rate, true_positive_rate, label=' '.join([model + ':', str(round(roc_auc, 3))]))

    # Add "Random" score
    plt.plot([0, 1], [0, 1], 'r--', label=' '.join(["Random:", str(.5)]))

    plt.legend(title='AUC Score', loc=4)
    #### Change filename for xG/xReb
    fig.savefig("ROC_AUC.png")

    


def fit_logistic(features_train, labels_train):
    """
    Fit the logistic regression and use cross validation to tune the hyperparameters

    :return: classifier
    """
    print("Fitting Logistic")

    param_grid = {
        'C': [.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    }

    clf = LogisticRegression(penalty='l2', solver='sag', random_state=42, max_iter=10000, tol=.01)

    # Tune hyperparameters
    cv_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10)

    # Fit classifier
    cv_clf.fit(features_train, labels_train)

    print("\nLogistic Regression Classifier:", cv_clf)

    # Save model
    pickle.dump(cv_clf, open("log_xg.pkl", 'wb'))

    return cv_clf

def fit_random_forest(features_train, labels_train):
    """
    Fit random forest and use cross validation to tune the hyperparameters

    :return: classifier
    """
    param_grid = {
        'min_samples_leaf': [50, 100, 250, 500]
    }

    clf = RandomForestClassifier(n_estimators=100, random_state=42, verbose=2)

    # Tune hyperparameters
    cv_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10)

    print("Fitting Random Forest")

    cv_clf.fit(features_train, labels_train)
    print("\nRandom Forest Classifier:", cv_clf)

    # Save model
    pickle.dump(cv_clf, open("random_forest_xg.pkl", 'wb'))
    return cv_clf

def fit_xgboost(features_train, labels_train):
    """
    Fit a gradient boosting algorithm and use cross validation to tune the hyperparameters

    :return: classifier
    """
    param_grid = {
        "gamma": [0, 0.25, 1],
        "reg_lambda": [0, 1, 10],
        "scale_pos_weight": [1, 3, 5],
        "subsample": [0.8],
        "colsample_bytree": [0.5],
        'max_depth': [3, 4, 5]
    }

    clf = XGBClassifier(n_estimators=500, eta=.1, random_state=42, verbosity=None)

    print("Fitting Gradient Boosting Classifier")

    # Tune hyperparameters
    cv_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10)

    # Fit classifier
    cv_clf.fit(features_train, labels_train)

    print("\nGradient Boosting Classifier:", cv_clf)

    # Save model
    pickle.dump(cv_clf, open("xgboost_xg.pkl", 'wb'))

    return cv_clf


def xg_model():
    """
    Create and test xg model.
    
    Fit models (Refer to those specific functions for more info):
    1. Logistic regression
    
    """
    data = clean_data(shots_sorted)
    
    data['Outcome'] = np.where(shots_sorted['Outcome'] == 0, 0, np.where(shots_sorted['Outcome'] == 1, 0, np.where(shots_sorted['Outcome'] == 2, 1, 3)))
    data = shots_sorted[shots_sorted['Outcome'] != 3]

    # Convert to lists
    features, labels = clean_data.convert_goals(data)

    # Split into training and testing sets -> 80/20
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=.2, random_state=42)

    # Fix Data
    features_train, labels_train = np.array(features_train), np.array(labels_train).ravel()
    
    ###### FIT MODELS
    log_clf = fit_logistic(features_train, labels_train)
    
    
    #### Testing
    log_preds_probs = log_clf.predict_proba(features_test)
    

    # Convert test labels to list instead of lists of lists
    flat_test_labels = [label[0] for label in labels_test]

    ### LOG LOSS
    print("\nLog Loss: ")
    print("Logistic Regression: ", log_loss(flat_test_labels, log_preds_probs))
    

    ### ROC
    preds = {
        "Logistic Regression": log_preds_probs,
        
    }
    get_roc(flat_test_labels, preds)

def main():
    xg_model()
    
if __name__ == '__main__':
    main()