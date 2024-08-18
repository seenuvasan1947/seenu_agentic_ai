class Task:
    def __init__(self, name, task_id, agent=None):
        self.name = name
        self.task_id = task_id
        self.agent = agent
    
    def run(self, inputs, mode='sequential'):
        if not self.agent:
            raise ValueError("No agent provided.")
        
        if mode == 'sequential':
            results = [self.agent.perform_task(input_text) for input_text in inputs]
            return results
        elif mode == 'hierarchical':
            result = inputs[0]
            for input_text in inputs:
                result = self.agent.perform_task(input_text)
            return [result]
        else:
            raise ValueError("Unsupported mode")

