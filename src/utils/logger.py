import logging
import os
import yaml

def setup_logger(config_path='config/config.yaml'):
    # Get the root logger instead of module-specific logger
    logger = logging.getLogger()
    
    # Check if logger is already configured
    if logger.handlers:
        return logger

    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logging_config = config['logging']
    
    # Create logs directory if not exists
    os.makedirs(os.path.dirname(logging_config['log_file']), exist_ok=True)
    
    logger.setLevel(logging_config['log_level'])
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(logging_config['log_file'])
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Clear any existing handlers
    logger.handlers = []
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger