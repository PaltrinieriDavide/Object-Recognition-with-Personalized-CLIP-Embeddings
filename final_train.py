import pandas as pd
import joblib
import numpy as np
import argparse
import os
import shutil
import logging

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate


from model_graphs import ModelGraphs

def show_graphs(model, X_train, y_train, X_test, y_test, output_dir = "graphs"):
    model_graphs = ModelGraphs(model, X_train, y_train, X_test, y_test, output_dir)
    model_graphs.generate_all_graphs()

def save_model(model, folder = "final_model"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    joblib.dump(model, folder + "/" + model.__class__.__name__ + ".pkl")

def prepocessing_phase(args):
    try:
        train_dataset = pd.read_csv(args.path_train)
        logging.debug("Dataset " + args.path_train + " opened correctly.")
    except Exception as e:
        logging.error(f"Failed to open training dataset at {args.path_train}: {e}")
        raise
    
    try:
        test_dataset = pd.read_csv(args.path_test)
        logging.debug("Dataset " + args.path_test + " opened correctly.")
    except Exception as e:
        logging.error(f"Failed to open training dataset at {args.path_test}: {e}")
        raise
    
    X_train = train_dataset.drop('y', axis=1)
    y_train = train_dataset['y']
    X_test = test_dataset.drop('y', axis=1)
    y_test = test_dataset['y']
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def calculate_metrics(y_true, y_pred):
    calculated_metrics={
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "conf_matrix": confusion_matrix(y_true, y_pred)
    }
    return calculated_metrics

def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path_train', help="Training dataset")
    parser.add_argument('path_test', nargs='?', help="Testing dataset", default=None)
    return parser.parse_args()


def plot_explained_variance(pca, output_dir="results"):
    import matplotlib.pyplot as plt
    # Calcola la varianza spiegata cumulativa
    varianza_spiegata_cumulativa = np.cumsum(pca.explained_variance_ratio_)
    
    # Traccia la varianza spiegata cumulativa
    plt.figure(figsize=(8, 6))
    plt.plot(varianza_spiegata_cumulativa, marker='o', linestyle='--')
    plt.title('Varianza cumulativa')
    plt.xlabel('Numero di componenti principali')
    plt.ylabel('Varianza cumulativa')
    plt.grid(True)
    plt.show()

    plt.savefig(os.path.join(output_dir, "explained_variance.png"))

def print_pca_loadings(pca, feature_names, model_name, num_components=6):
    loadings = pca.components_
    explained_variance = pca.explained_variance_ratio_
    
    print(f"\nTop 10 contributing features for {model_name} PCA loadings (for the first {num_components} components):\n")
    for i, component in enumerate(loadings[:num_components]):
        print(f"Component {i + 1} (explains {explained_variance[i]:.2%} of the variance):")
        
        # Ordina le features in base al valore assoluto dei loadings
        loading_abs_sorted = sorted(zip(feature_names, component), key=lambda x: abs(x[1]), reverse=True)
        
        # Stampa solo le prime 10 features
        for feature, loading in loading_abs_sorted[:10]:
            print(f"{feature}: {loading:.4f}")
        print("\n")
        
def generate_feature_names(n = 77):
    """This function generates a list of names sim1, sim2, ..., simN"""
    return [f"sim{i+1}" for i in range(n)]


def main(models):
    
    results_folder = "results"
    if os.path.exists(results_folder):
        shutil.rmtree(results_folder)
    os.makedirs(results_folder)
    logging.basicConfig(filename= os.path.join(results_folder, 'data.log'), filemode='w', level=logging.DEBUG, format='%(message)s')
    logging.getLogger('matplotlib.font_manager').disabled = True

    args = parse_command_line_args()
    
    X_train, X_test, y_train, y_test = prepocessing_phase(args)
    
    cv = StratifiedKFold(n_splits=7, shuffle=True)
    # feature_selector = SelectKBest(score_func=chi2, k=30)
    feature_selector = PCA(n_components=0.99)
    feature_selector.fit(X_train)
    plot_explained_variance(feature_selector)


    grid_search_results = {}

    best_model_name = None
    best_score = 0
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1': make_scorer(f1_score, average='weighted')
    }
    
    for model_name, model_info in models.items():
        print(model_name)
        pipeline = Pipeline([
            ('model', model_info['model'])
        ])
        
        #without feature selection
        
        #grid search
        grid_search = GridSearchCV(pipeline, model_info['params'], cv=cv, n_jobs=-1, scoring=scoring, refit='accuracy')
        grid_search.fit(X_train, y_train)
        
        #cross validation
        cv_scores = cross_validate(grid_search.best_estimator_, X_train, y_train, cv=cv, scoring=scoring)
        # cross_results[model_name] = np.mean(cv_scores)
        
        grid_search_results[model_name] = grid_search
        
        if np.mean(cv_scores['test_accuracy']) > best_score:
            best_score = np.mean(cv_scores['test_accuracy'])
            best_model_name = model_name
        
        logging.debug("Model name: " + model_name)
        logging.debug(f"Best parameters: {grid_search.best_params_}")
        logging.debug(f"Mean CV accuracy: {np.mean(cv_scores['test_accuracy']):.4f}")
        logging.debug(f"Mean CV precision: {np.mean(cv_scores['test_precision']):.4f}")
        logging.debug(f"Mean CV recall: {np.mean(cv_scores['test_recall']):.4f}")
        logging.debug(f"Mean CV F1 Score: {np.mean(cv_scores['test_f1']):.4f}")
        logging.debug("\n" + "-" * 30 + "\n")
        
        
        #with feature selection
        
        # pipeline = Pipeline([
        #     ('feature_selection', feature_selector),
        #     ('model', model_info['model'])
        # ])
        
        pipeline = Pipeline([
             ('pca', feature_selector),
             ('model', model_info['model'])
         ])
        
        #grid search
        grid_search = GridSearchCV(pipeline, model_info['params'], cv=cv, n_jobs=-1, scoring=scoring, refit='accuracy')
        grid_search.fit(X_train, y_train)
        
        #cross validation
        cv_scores = cross_validate(grid_search.best_estimator_, X_train, y_train, cv=cv, scoring=scoring)
        # cross_results[model_name] = np.mean(cv_scores)
        
        if np.mean(cv_scores['test_accuracy']) > best_score:
            best_score = np.mean(cv_scores['test_accuracy'])
            best_model_name = model_name + "_fs"
        
        grid_search_results[model_name + "_fs"] = grid_search
        
        logging.debug("Model name: " + model_name + "_fs")
        logging.debug(f"Best parameters: {grid_search.best_params_}")
        logging.debug(f"Mean CV accuracy: {np.mean(cv_scores['test_accuracy']):.4f}")
        logging.debug(f"Mean CV precision: {np.mean(cv_scores['test_precision']):.4f}")
        logging.debug(f"Mean CV recall: {np.mean(cv_scores['test_recall']):.4f}")
        logging.debug(f"Mean CV F1 Score: {np.mean(cv_scores['test_f1']):.4f}")
        logging.debug("\n" + "-" * 30 + "\n")
        
        if 'feature_selection' in grid_search.best_estimator_.named_steps:
            selected_features = grid_search.best_estimator_.named_steps['feature_selection'].get_support(indices=True)
            logging.debug(f"\nSelected features for {model_name} with feature selection: {selected_features}")
        
        feature_names = generate_feature_names()
        if 'pca' in grid_search.best_estimator_.named_steps:
            pca = grid_search.best_estimator_.named_steps['pca']
            print(f"\nPCA Loadings for {model_name}:\n")
            print_pca_loadings(pca, feature_names, model_name, num_components=7) 
                
    
    best_model = grid_search_results[best_model_name].best_estimator_
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    
    logging.debug("# " * 125)
    logging.debug(f"\nMiglior Modello: {best_model_name}")
    if "_fs" in best_model_name:
        logging.debug(f"\nThe model uses Feature Selection")
    else:
        logging.debug(f"\nThe model does not use Feature Selection")
    logging.debug(f"Best Cross-Validation Accuracy: {best_score}")
    logging.debug(f"Valutazione finale sul set di test:")
    logging.debug(f"Metrics: {calculate_metrics(y_test, y_pred)}")
    logging.debug(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    
    # show_graphs(best_model, X_train, y_train, X_test, y_test, os.path.join(results_folder, "graphs"))
    save_model(best_model, os.path.join(results_folder, "final_model"))
    
    model_graphs = ModelGraphs(y_train, y_test, y_pred, best_model.predict_proba(X_test), os.path.join(results_folder, "graphs"), ["0", "1"])
    model_graphs.generate_all_reports()

if __name__ == '__main__':
    models = {
        'LogisticRegression': {
            'model': LogisticRegression(),
            'params': {
                'model__C': [0.01, 0.1, 1, 10],
                'model__penalty': ['l2', 'l1'],
                'model__solver': ['liblinear', 'saga']
            }
        },
        'KNeighborsClassifier': {
            'model': KNeighborsClassifier(),
            'params': {
                'model__n_neighbors': [3, 5, 7, 9],
                'model__weights': ['uniform', 'distance']
            }
        },
        'SVC': {
            'model': SVC(probability=True),
            'params': {
                'model__C': [0.1, 1, 10],
                'model__kernel': ['linear', 'rbf', 'poly'],
                'model__gamma': ['scale', 'auto']
            }
        },
        'DecisionTreeClassifier': {
            'model': DecisionTreeClassifier(),
            'params': {
                'model__max_depth': [None, 10, 20],
                'model__min_samples_split': [2, 5],
                'model__criterion': ['gini', 'entropy']
            }
        },
        'RandomForestClassifier': {
            'model': RandomForestClassifier(),
            'params': {
                'model__n_estimators': [50, 100, 200],
                'model__min_samples_split': [2, 5, 10],
                'model__max_depth': [None, 10, 20]
            }
        }
    }
    
    main(models)