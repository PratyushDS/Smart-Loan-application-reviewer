import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import yaml
from src.utils.logger import setup_logger
import logging
logger = logging.getLogger(__name__)

class EDA:
    def __init__(self, config_path='config/config.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.reports_config = self.config['reports']
        os.makedirs(self.reports_config['figures_dir'], exist_ok=True)
        os.makedirs(self.reports_config['tables_dir'], exist_ok=True)
        
    def load_data(self):
        try:
            df = pd.read_csv(self.config['data_path'])
            logger.info("Data loaded successfully")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def perform_eda(self):
        df = self.load_data()
        
        # Generate basic statistics
        self._generate_summary_stats(df)
        
        # Visualizations
        self._plot_target_distribution(df)
        self._plot_missing_values(df)
        self._plot_numerical_distributions(df)
        self._plot_categorical_distributions(df)
        self._plot_correlation_matrix(df)
        
        # Save value counts
        self._save_value_counts(df)
        
        logger.info("EDA completed successfully")
    
    def _generate_summary_stats(self, df):
        # Basic statistics
        stats = df.describe(include='all').T
        stats.to_excel(os.path.join(self.reports_config['tables_dir'], 'summary_stats.xlsx'))
        
        # Missing values
        missing = pd.DataFrame(df.isnull().sum(), columns=['missing_count'])
        missing['missing_pct'] = (missing['missing_count'] / len(df)) * 100
        missing.to_excel(os.path.join(self.reports_config['tables_dir'], 'missing_values.xlsx'))
    
    def _plot_target_distribution(self, df):
        plt.figure(figsize=(8, 6))
        df[self.config['target']].value_counts().plot(kind='bar')
        plt.title('Target Variable Distribution')
        plt.savefig(os.path.join(self.reports_config['figures_dir'], 'target_distribution.png'))
        plt.close()
    
    def _plot_missing_values(self, df):
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False)
        plt.title('Missing Values Heatmap')
        plt.savefig(os.path.join(self.reports_config['figures_dir'], 'missing_values.png'))
        plt.close()
    
    def _plot_numerical_distributions(self, df):
        numerical = self.config['features']['numerical']
        for col in numerical:
            plt.figure(figsize=(8, 6))
            sns.histplot(df[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.savefig(os.path.join(self.reports_config['figures_dir'], f'{col}_dist.png'))
            plt.close()
    
    def _plot_categorical_distributions(self, df):
        categorical = self.config['features']['categorical']
        for col in categorical:
            plt.figure(figsize=(8, 6))
            df[col].value_counts().plot(kind='pie', autopct='%1.1f%%')
            plt.title(f'Distribution of {col}')
            plt.savefig(os.path.join(self.reports_config['figures_dir'], f'{col}_dist.png'))
            plt.close()
    
    def _plot_correlation_matrix(self, df):
        plt.figure(figsize=(12, 10))
        corr = df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.savefig(os.path.join(self.reports_config['figures_dir'], 'correlation_matrix.png'))
        plt.close()
    
    def _save_value_counts(self, df):
        categorical = self.config['features']['categorical']
        with pd.ExcelWriter(os.path.join(self.reports_config['tables_dir'], 'value_counts.xlsx')) as writer:
            for col in categorical:
                df[col].value_counts().to_excel(writer, sheet_name=col[:31])