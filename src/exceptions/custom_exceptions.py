#this is a custom exceptions hierarchy for the mlops project
#handles all pipeline-specific exceptions - they inherit from this pipelineexception class

class PipelineException(Exception):
    """base excption for all pipeline errors"""
    def __init__(self,message,original_exception=None):
        super().__init__(message)
        self.original_exception = original_exception
        self.message = message

    def __str__(self):
        #allows proper display of error strings
        if self.original_exception:
            return f"{self.message} (caused by: {repr(self.original_exception)})"
        return self.message
    
class ConfigurationError(PipelineException):
    """this is raised when the config.yaml file is missing, malformed
    or contains invalid values
    """
    pass
class DataIngestionError(PipelineException):
    """this is raised when raw data cannot be loaded (file missing, wrong format)"""
    pass

class PreprocessingError(PipelineException):
    """it is raised during the preprocessing stage of the pipeline"""
    pass

class ModelTrainigError(PipelineException):
    """this is raised during model fitting and there are issues"""
    pass
class ModelEvaluationError(PipelineException):
    """raised when metrics computation or cross-validation fails"""
    pass
class MLflowRegistrationError(PipelineException):
    """raised when mlflow tracking or model registration fails"""
    pass
class InferenceError(PipelineException):
    """raised during the inference of the model"""
    pass
class DataDownloadError(PipelineException):
    """raised when there are issues downloading data from external sources"""
    pass
class ModelRegistrationError(PipelineException):
    """raised when there is an error in registring the model"""
    pass