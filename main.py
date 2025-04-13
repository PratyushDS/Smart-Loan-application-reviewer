import sys
from pathlib import Path
from src.visualization.eda import EDA
from src.utils.logger import setup_logger
from src.data_preprocessor import DataPreprocessor
from src.model_training import ModelTrainer

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

def training_pipeline():
    # Setup logging
    logger = setup_logger()
    
    try:
        # Perform EDA
        logger.info("Starting EDA")
        eda = EDA()
        eda.perform_eda()
        
        # Data preparation
        # logger.info("Preparing data")
        # processor = DataPreprocessor('config/config.yaml')
        # X_train, X_test, y_train, y_test, feature_names = processor.prepare_data()
        
        # # Model training and evaluation
        # logger.info("Starting model training")
        # trainer = ModelTrainer('config/config.yaml')
        # trainer.train_and_evaluate(X_train, y_train, X_test, y_test, feature_names)
        # print("Model training completed")
        # logger.info("Model training completed")
    except Exception as e:
        logger.error(f"Error in main pipeline: {str(e)}")
        raise


if __name__ == '__main__':
    training_pipeline()
    print("Pipeline completed successfully")