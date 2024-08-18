class TaskManager:
    def __init__(self):
        self.tasks = []

    def add_task(self, task):
        self.tasks.append(task)

    def run(self, inputs, mode='sequential'):
        results = inputs
        if mode == 'sequential':
            all_results = []
            for task in self.tasks:
                task_results = task.run(results, mode='sequential')
                all_results.append(task_results)
                results = task_results  # Pass results to the next task
            return all_results
        elif mode == 'hierarchical':
            for task in self.tasks:
                if isinstance(results, list):
                    results = results[0] if len(results) == 1 else results
                if isinstance(results, dict):
                    results = next(iter(results.values()))
                results = task.run([results], mode='hierarchical')[0]
            return [results]
        else:
            raise ValueError("Unsupported mode")

