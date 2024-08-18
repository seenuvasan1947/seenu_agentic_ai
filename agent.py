from transformers import pipeline

class Agent:
    def __init__(self, name, llm=None, function=None):
        self.name = name
        self.llm = llm  # llm is a dictionary with 'repo_name' and 'operation'
        self.function = function
        self.model = None
        if llm:
            self.initialize_model()
    
    def initialize_model(self):
        repo_name = self.llm['repo_name']
        operation = self.llm['operation']
        # Create a Hugging Face pipeline for the specified operation
        self.model = pipeline(operation, model=repo_name)
    
    def perform_task(self, input_text):
        if self.model:
            result = self.model(input_text)
            if isinstance(result, list) and isinstance(result[0], dict):
                # Generic handling of various tasks
                return next(iter(result[0].values()))  # Get the first value in the dictionary
            return result
        elif self.function:
            return self.function(input_text)
        else:
            raise ValueError("No model or function provided.")

