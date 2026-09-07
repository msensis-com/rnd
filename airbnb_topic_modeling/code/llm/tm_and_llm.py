import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from services.topic_data import get_topics
import time

load_dotenv()

endpoint = "https://flexbot-ai-services.services.ai.azure.com/models"
model_name = "AI21-Jamba-1.5-Large"


def send_message(system_message, user_message):
    client = ChatCompletionsClient(
        endpoint,
        AzureKeyCredential(os.environ["AZURE_KEY"]),
        api_version="2024-05-01-preview"
    )

    response = client.complete(
        messages=[
            SystemMessage(content=system_message),
            UserMessage(content=user_message),
        ],
        max_tokens=1000,
        temperature=0.8,
        top_p=0.5,
        seed=42,
        model=model_name
    )
    try:
        return response["choices"][0]["message"]["content"]
    except:
        return response


if __name__ == '__main__':
    start_time = time.time()

    system_prompt = """
    Extract a topic title or brief description based on the given keywords. Write at most a sentence per given topic keywords. Output format MUST be <Index Number of Topic>. <Topic Title>.
    """

    t2v, btopic = get_topics()
    
    counter = 0
    topics = 'Index Number of Topic, Topic Keywords\n'
    for topic in t2v:
        counter += 1
        topics += f'{counter}. {topic}\n'
    
    for topic in btopic:
        counter += 1
        topics += f'{counter}. {topic}\n'

    results = send_message(system_prompt, topics)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Results: {results}")
    with open (f"./results/256k_dataset/{index_of_data_part}.txt", 'a+') as f:
        f.write(f"{results}\nTime: {elapsed_time}")
    f.close()
    
