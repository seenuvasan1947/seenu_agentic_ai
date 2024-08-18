from transformers import pipeline

# Define the Agent class
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

# Define the Task class
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

# Define the TaskManager class to handle multiple tasks
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

# Example usage

# Define the translation agent using an LLM
translation_llm = {'repo_name': 'Helsinki-NLP/opus-tatoeba-fi-en', 'operation': 'translation'}
translation_agent = Agent(name='Translator', llm=translation_llm)

# Define the summarization agent using an LLM
summarization_llm = {'repo_name': 'facebook/bart-large-cnn', 'operation': 'summarization'}
summarization_agent = Agent(name='Summarizer', llm=summarization_llm)

# Define the classification agent using an LLM
classification_llm = {'repo_name': 'distilbert-base-uncased-finetuned-sst-2-english', 'operation': 'sentiment-analysis'}
classification_agent = Agent(name='Classifier', llm=classification_llm)

# Define a custom function
def custom_function(text):
    return text.upper()
def custom_function_write(text):
    with open('custom_output.txt', 'w') as f:
        f.write(text)
    return "Task completed sucessfully"    

# Define an agent using a custom function
custom_agent = Agent(name='Custom Function Agent', function=custom_function)
custom_agent_write = Agent(name='Custom Function Agent write', function=custom_function_write)

# Define tasks
translation_task = Task(name='Translation Task', task_id=1, agent=translation_agent)
summarize_task = Task(name='Summarize Task', task_id=2, agent=summarization_agent)
classify_task = Task(name='Classify Task', task_id=3, agent=classification_agent)
custom_task = Task(name='Custom Function Task', task_id=4, agent=custom_agent)
custom_task_write = Task(name='Custom Function Task write', task_id=5, agent=custom_agent_write)

# Add tasks to TaskManager
task_manager = TaskManager()
task_manager.add_task(translation_task)
task_manager.add_task(summarize_task)
task_manager.add_task(classify_task)
task_manager.add_task(custom_task)
task_manager.add_task(custom_task_write)

# Input data
input_texts = [
    """Intia, maa, jossa on runsaasti historiaa, kulttuuria ja monimuotoisuutta, on maailman suurin demokratia ja yksi nopeimmin kasvavista suurista talouksista. 
    Intia on kontrastien ja monimutkaisuuden maa, joka ulottuu pohjoisen lumihuippuisista Himalajasta etelän trooppisiin rantoihin. Sen yli 1,4 miljardin asukkaan väkiluku käsittää lukuisia kieliä, 
    uskontoja ja perinteitä, mikä tekee siitä elävän kulttuurisen monimuotoisuuden mosaiikin. 
    Intian talous, jota ohjaavat vahva palvelusektori, kukoistava teknologiateollisuus ja kasvava tuotantopohja, on kehittynyt nopeasti viime vuosikymmeninä. 
    Silti maa kohtaa myös haasteita, kuten tuloerot, ympäristön rappeutuminen ja infrastruktuurin kehittämisen tarve.

Intian historiaa leimaa joukko voimakkaita imperiumia Mauryoista ja Guptaista Mughaleihin ja British Rajiin, 
joista jokainen edistää sen rikasta kulttuuri- ja arkkitehtuuriperintöä. Itsenäistymisen jälkeen Intia on säilyttänyt vahvan demokraattisen kehyksen, ja perustuslaki sisältää maallistumisen, 
sosiaalisen oikeudenmukaisuuden ja tasa-arvon. Maa tunnetaan myös panoksestaan ​​taiteeseen, tieteeseen ja henkisyyteen, 
muinaisista Veda- ja Upanishad-teksteistä nykyajan saavutuksiin avaruustutkimuksessa ja tietotekniikassa.

Moderni Intia on johtaja globaalilla areenalla, joka tasapainottaa syvälle juurtuneet perinteensä ja huippuluokan innovaatiot. Kaupungit, kuten Mumbai, Delhi ja Bangalore, ovat vilkkaita kaupan, 
kulttuurin ja teknologian keskuksia, kun taas maaseutualueet ovat edelleen perinteisen intialaisen elämän sydän. 
Diwalin, Holin ja Eidin kaltaisia ​​festivaaleja juhlitaan valtavasti innostuneesti kaikkialla maassa, mikä kuvastaa Intiaa määrittelevää yhtenäisyyttä monimuotoisuudessa. 
Kun kansakunta jatkaa kehittymistään, se on edelleen syvästi yhteydessä menneisyyteensä ja odottaa kasvua, kehitystä ja globaalia johtajuutta."""
]

# Run tasks sequentially
print("Sequential Task Output:")
sequential_results = task_manager.run(input_texts, mode='sequential')
for i, task_results in enumerate(sequential_results):
    print(f"Task {i+1} Results:")
    for j, result in enumerate(task_results):
        print(f"  Result : {result}")

# Run tasks in hierarchical mode
print("\nHierarchical Task Output:")
hierarchical_results = task_manager.run(input_texts, mode='hierarchical')
for i, result in enumerate(hierarchical_results):
    print(f"Final Result : {result}")

