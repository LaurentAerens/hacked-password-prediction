import logging

class LoggerWrapper:
    def __init__(self, logger):
        self.logger = logger

    def info(self, msg):
        if self.logger:
            self.logger.info(msg)

    def warning(self, msg):
        if self.logger:
            self.logger.warning(msg)

    def error(self, msg):
        if self.logger:
            self.logger.error(msg)

def create_logger():
    """
    Configure and create a logger.
    
    Returns:
    tuple: (LoggerWrapper, str) - Configured logger instance wrapped in LoggerWrapper and a warning message if any.
    """
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("genetic_ai_trainer.log"),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger(__name__)
        return LoggerWrapper(logger), None
    except Exception as e:
        # If logger creation fails, return a LoggerWrapper with None and a warning message
        warning_msg = f"Warning: Failed to create logger. Error: {e}"
        return LoggerWrapper(None), warning_msg