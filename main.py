from agent import Agent
from task import Task
from task_manager import TaskManager

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
    return "Task completed successfully"    

# Define an agent using a custom function
custom_agent = Agent(name='Custom Function Agent', function=custom_function)
custom_agent_write = Agent(name='Custom Function Agent Write', function=custom_function_write)

# Define tasks
translation_task = Task(name='Translation Task', task_id=1, agent=translation_agent)
summarize_task = Task(name='Summarize Task', task_id=2, agent=summarization_agent)
classify_task = Task(name='Classify Task', task_id=3, agent=classification_agent)
custom_task = Task(name='Custom Function Task', task_id=4, agent=custom_agent)
custom_task_write = Task(name='Custom Function Task Write', task_id=5, agent=custom_agent_write)

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
    Silti maa kohtaa myös haasteita, kuten tuloerot, ympäristön rappeutuminen ja infrastruktuurin kehittämisen tarve."""
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

