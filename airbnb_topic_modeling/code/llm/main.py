import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from services.read_data_service import create_str_from_data
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
        model=model_name
    )
    try:
        return response["choices"][0]["message"]["content"]
    except:
        return response


if __name__ == '__main__':
    start_time = time.time()

    system_prompt = """
    Analyze the provided content and identify the top 10 negative most frequently discussed or prominent topics across the entire dataset.
Read all text entries in the dataset. You must detect and group similar themes, keywords, or recurring subjects you must rank the topics by relevance or frequency and then output a numbered list of the top 10 topics, each with a description using 2-10 words explaining what the topic is about.you must focus on topics, not individual keywords (e.g., "customer service quality" instead of just "service"). Avoid duplicating similar topics and merge them if needed, do not include any personal data. Output also and a score per topic that represents the number of lines in content that belong to each topic. Output plain text, no markdowns! For each topic output a Title, a brief description and the score of how many lines belong to it.
    """
    base_datapath_1 = "./data/128k_dataset/new_clear_data_part_"
    base_datapath_2 = "./data/256k_dataset/new_clear_data_part_"
    index_of_data_part = "5"
    user_prompt = create_str_from_data(f"{base_datapath_2}{index_of_data_part}.csv")
    results = send_message(system_prompt, user_prompt)
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Results: {results}")
    with open (f"./results/256k_dataset/{index_of_data_part}.txt", 'a+') as f:
        f.write(f"{results}\nTime: {elapsed_time}")
    f.close()
