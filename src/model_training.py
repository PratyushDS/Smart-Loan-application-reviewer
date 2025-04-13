from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score,confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
import yaml
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime
from src.utils.logger import setup_logger
import logging
logger = logging.getLogger(__name__)

class CustomXGBClassifier(BaseEstimator, ClassifierMixin):
    """XGBoost classifier wrapper for scikit-learn compatibility"""
    def __init__(self, random_state=None, n_estimators=100, max_depth=3, learning_rate=0.1):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self._estimator = XGBClassifier(
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
    def fit(self, X, y):
        self._estimator.fit(X, y)
        return self
        
    def predict(self, X):
        return self._estimator.predict(X)
        
    def predict_proba(self, X):
        return self._estimator.predict_proba(X)
        
    def get_params(self, deep=True):
        return {
            'random_state': self.random_state,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate
        }
        
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        self._estimator.set_params(**parameters)
        return self

    @property
    def feature_importances_(self):
        return self._estimator.feature_importances_

class ModelTrainer:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.classifiers = [
            {
                'name': 'Logistic Regression',
                'model': LogisticRegression(random_state=self.config['random_state'], max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'solver': ['liblinear', 'saga']
                }
            },
            {
                'name': 'Naive Bayes',
                'model': GaussianNB(),
                'params': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7]
                }
            },
            {
                'name': 'SVM',
                'model': SVC(random_state=self.config['random_state'], probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'rbf']
                }
            },
            {
                'name': 'Decision Tree',
                'model': DecisionTreeClassifier(random_state=self.config['random_state']),
                'params': {
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                }
            },
            {
                'name': 'Random Forest',
                'model': RandomForestClassifier(random_state=self.config['random_state']),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5]
                }
            },
            {
                'name': 'Gradient Boosting',
                'model': GradientBoostingClassifier(random_state=self.config['random_state']),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                }
            },
            {
                'name': 'XGBoost',
                'model': CustomXGBClassifier(random_state=self.config['random_state']),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5],
                    'learning_rate': [0.01, 0.1]
                }
            },
            {
                'name': 'CatBoost',
                'model': CatBoostClassifier(random_state=self.config['random_state'], silent=True),
                'params': {
                    'iterations': [100, 200],
                    'depth': [3, 5],
                    'learning_rate': [0.01, 0.1]
                }
            },
            {
                'name': 'LightGBM',
                'model': LGBMClassifier(random_state=self.config['random_state']),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5],
                    'learning_rate': [0.01, 0.1]
                }
            }
        ]
        
        self.reports_config = self.config['reports']
        os.makedirs(self.reports_config['tables_dir'], exist_ok=True)
        
        self.best_model = None
        self.best_score = -np.inf
        self.best_model_name = ''
        self.results = []

    def train_and_evaluate(self, X_train, y_train, X_test, y_test, feature_names):
        logger.info("Starting model training and evaluation process")
        
        try:
            for classifier in self.classifiers:
                logger.info(f"Training {classifier['name']} started")
                start_time = datetime.now()
                
                grid_search = GridSearchCV(
                    estimator=classifier['model'],
                    param_grid=classifier['params'],
                    scoring='f1',
                    cv=5,
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_
                
                # Calculate metrics for both sets
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)
                
                train_metrics = self._calculate_metrics(y_train, y_train_pred)
                test_metrics = self._calculate_metrics(y_test, y_test_pred)
                
                # Store comprehensive  results
                self.results.append({
                'model': classifier['name'],
                'best_params': grid_search.best_params_,
                # Training metrics
                'train_f1': train_metrics['f1'],
                'train_precision': train_metrics['precision'],
                'train_recall': train_metrics['recall'],
                'train_accuracy': train_metrics['accuracy'],
                'train_tp': train_metrics['tp'],
                'train_fp': train_metrics['fp'],
                'train_tn': train_metrics['tn'],
                'train_fn': train_metrics['fn'],
                # Test metrics
                'test_f1': test_metrics['f1'],
                'test_precision': test_metrics['precision'],
                'test_recall': test_metrics['recall'],
                'test_accuracy': test_metrics['accuracy'],
                'test_tp': test_metrics['tp'],
                'test_fp': test_metrics['fp'],
                'test_tn': test_metrics['tn'],
                'test_fn': test_metrics['fn'],
                'training_time': str(datetime.now() - start_time)
            })
                
                logger.info(f"{classifier['name']} training completed in {self.results[-1]['training_time']}")
                logger.info(f"{classifier['name']} training Metrics - F1: {train_metrics['f1']:.4f}, "
                          f"Precision: {train_metrics['precision']:.4f}, "
                          f"Recall: {train_metrics['recall']:.4f}, "
                          f"Accuracy: {train_metrics['accuracy']:.4f}")
                
                logger.info(f"{classifier['name']} test Metrics - F1: {test_metrics['f1']:.4f}, "
                          f"Precision: {test_metrics['precision']:.4f}, "
                          f"Recall: {test_metrics['recall']:.4f}, "
                          f"Accuracy: {test_metrics['accuracy']:.4f}")
                
                if test_metrics['f1'] > self.best_score:
                    self.best_score = test_metrics['f1']
                    self.best_model = best_model
                    self.best_model_name = classifier['name']
            
            # Save results and models
            self._save_results()
            self._save_best_model()
            self._save_feature_importance(feature_names)
            
            logger.info(f"Best model: {self.best_model_name} with F1 score: {self.best_score:.4f}")
            return self.best_model
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise

    def _calculate_metrics(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return {
            'f1': f1_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'accuracy': accuracy_score(y_true, y_pred),
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        }

    def _save_results(self):
        try:
            # Save model comparison results
            df = pd.DataFrame(self.results)
            df.sort_values('test_f1', ascending=False, inplace=True)
            
            path = os.path.join(self.reports_config['tables_dir'], 'model_performance.xlsx')
            df.to_excel(path, index=False)
            logger.info(f"Model comparison results saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

    def _save_best_model(self):
        try:
            joblib.dump(self.best_model, self.config['model_path'])
            
            logger.info(f"Best model ({self.best_model_name}) saved to {self.config['model_path']}")
        except Exception as e:
            logger.error(f"Error saving best model: {str(e)}")
            raise
    
    # Add this method to ModelTrainer class
    def _validate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        
        st.write("## Model Validation Report")
        st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
        st.write(f"**F1 Score:** {f1_score(y_test, y_pred):.2f}")
        st.write(f"**Class Distribution:**")
        st.write(pd.Series(y_test).value_counts().to_frame('Count'))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        st.pyplot(fig)

    def _save_feature_importance(self, feature_names):
        """Save feature importance/coefficients for supported models"""
        try:
            # For tree-based models with feature_importances_
            if hasattr(self.best_model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': self.best_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                path = os.path.join(self.reports_config['tables_dir'], 
                                'feature_importance.xlsx')
                importance.to_excel(path, index=False)
                logger.info(f"Feature importance saved to {path}")
            
            # For linear models with coefficients
            elif hasattr(self.best_model, 'coef_'):
                importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': self.best_model.coef_[0]
                }).sort_values('Coefficient', key=abs, ascending=False)
                
                path = os.path.join(self.reports_config['tables_dir'], 
                                'feature_coefficients.xlsx')
                importance.to_excel(path, index=False)
                logger.info(f"Feature coefficients saved to {path}")
            
            # For models without importance/coefficients
            else:
                logger.warning(f"{self.best_model_name} doesn't support feature importance or coefficients")
                return
                
        except Exception as e:
            logger.error(f"Error saving feature importance: {str(e)}")
            raise