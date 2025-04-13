import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from src.utils.logger import setup_logger
import logging
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.preprocessor = None
        
    def load_data(self):
        try:
            df = pd.read_csv(self.config['data_path'])
            logger.info("Data loaded successfully")
            
            # Clean all string columns: remove leading/trailing whitespace
            str_cols = df.select_dtypes('object').columns
            df[str_cols] = df[str_cols].apply(lambda x: x.str.strip())
            
            # Clean target column specifically
            df[self.config['target']] = df[self.config['target']].str.strip().str.title()
            
            # Validate target values
            valid_values = ['Approved', 'Rejected']
            invalid_values = df[self.config['target']][~df[self.config['target']].isin(valid_values)].unique()
            
            if len(invalid_values) > 0:
                error_msg = f"Invalid values in target column: {invalid_values}"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            # Convert to binary
            df[self.config['target']] = df[self.config['target']].replace({
                'Approved': 1,
                'Rejected': 0
            })
            
            X = df.drop(self.config['target'], axis=1)
            y = df[self.config['target']]
            return X, y
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def get_preprocessor(self):
        try:
            numerical_features = self.config['features']['numerical']
            categorical_features = self.config['features']['categorical']
            
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ])
            return self.preprocessor
            
        except Exception as e:
            logger.error(f"Error creating preprocessor: {str(e)}")
            raise
    
    def prepare_data(self):
        try:
            X, y = self.load_data()
            
            # Split first before any preprocessing
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config['test_size'],
                random_state=self.config['random_state'],
                stratify=y
            )
            
            # Fit preprocessor ONLY on training data
            preprocessor = self.get_preprocessor()
            preprocessor.fit(X_train_raw)
            
            # Transform both sets using training parameters
            X_train = preprocessor.transform(X_train_raw)
            X_test = preprocessor.transform(X_test_raw)
            
            # Get feature names (from training data)
            numerical = self.config['features']['numerical']
            categorical = self.config['features']['categorical']
            cat_names = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical)
            self.feature_names = np.concatenate([numerical, cat_names])
            
            joblib.dump(preprocessor, self.config['preprocessor_path'])
            
            return X_train, X_test, y_train, y_test, self.feature_names
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise