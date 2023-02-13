import argparse
import os
import pickle
import time
from typing import Union

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC

from dataset import CarDetectorDataset
from utils import date_to_string

Classifier = Union[LinearSVC, GradientBoostingClassifier, RandomForestClassifier]

def train_classifier(model:str, X_train:np.ndarray, X_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray, **kwargs) -> Classifier:
    """Trains a given model, previously selected via cross-validation, on the whole dataset.

    Args:
        model (str): Classifier to train.
        X_train (np.ndarray): Training split.
        X_test (np.ndarray): Test split.
        y_train (np.ndarray): Training target class.
        y_test (np.ndarray): Test target class.

    Returns:
        Classifier: Fitted model on the available data.
    """
    assert model in ('linear_svm', 'gradient_boosting'), "Unsupported classifier !"
    match model:
        case 'linear_svm':
            clf = LinearSVC(C=1e-4, loss="squared_hinge", penalty="l2", dual=False, fit_intercept=False, random_state=42)
        case 'gradient_boosting':
            clf = GradientBoostingClassifier(n_estimators=300, random_state=42)

    # We train here our model on all available data:
    if X_test is not None:
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
    else:
        X, y = X_train, y_train
    
    # Training
    start_time = time.time()
    clf.fit(X, y)
    print(f"Classifier trained in {time.time() - start_time:.1f}")

    # Saving classifier
    filename = f"{kwargs['date']}_{kwargs['model_name']}.pkl"
    if not os.path.exists("models"):
        os.makedirs("models")
    with open(os.path.join("models", filename),"wb") as f:
        pickle.dump(clf, f)
    print(f"Model {filename} saved to models/")
    return clf



if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(
        description="Launches training."
    )
    parser.add_argument("--model_name", type=str, help="Name of the trained model")
    parser.add_argument('--bow', dest='bow', action='store_true')
    parser.add_argument('--no-bow', dest='bow', action='store_false')
    parser.set_defaults(bow=False)
    parser.add_argument('--spatial', dest='spatial', action='store_true')
    parser.add_argument('--no-spatial', dest='spatial', action='store_false')
    parser.set_defaults(spatial=False)
    parser.add_argument('--hist', dest='hist', action='store_true')
    parser.add_argument('--no-hist', dest='bow', action='store_false')
    parser.set_defaults(hist=False)
    
    args = parser.parse_args()
    
    date = date_to_string()
    
    kwargs = {
        'model_name':args.model_name,
        'date': date,   
    }

    # Reads dataframe
    df_data = pd.read_csv('./train.csv')   
    
    # Creates dataset and splits data
    dataset = CarDetectorDataset(
        df_data=df_data, 
        min_h=30, 
        min_w=30,
        date=date,
        window_size=64,
        bag_of_words=args.bow,
        hist=args.hist, 
        spatial=args.spatial,
        )
    X_train, X_test, y_train, y_test = dataset.create_dataset()

    # Launches training
    clf = train_classifier(X_train, X_test, y_train, y_test, **kwargs)
    
