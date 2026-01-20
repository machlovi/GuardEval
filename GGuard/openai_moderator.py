import csv
import re

from openai import OpenAI
from datasets import load_from_disk, DatasetDict
from tqdm import tqdm
from config import OPENAI_API_KEY
import os
# client = OpenAI()
client = OpenAI(api_key=OPENAI_API_KEY)
import openai

CHECKPOINT_FILE = "/home/naseem_fordham/LLM_evaluation/Moderators/Chatbase Moderators/Catplus_results/checkpoint.txt"

def check_moderator(text):
    response = openai.Moderation.create(input=text)
    return response

def save_checkpoint(index):
    with open(CHECKPOINT_FILE, "w") as f:
        f.write(str(index))

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return int(f.read().strip())
    return 0


def check_moderator(input_text):
    # Call the OpenAI moderation API
    response = client.moderations.create(
        model="omni-moderation-latest",
        input=input_text,
    )
    return response

# Helper functions
def clean_text(text):
    """Remove URLs, usernames, and extra spaces."""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@[^\s]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"\^[^ ]+", "", text)

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk, Sequence, Value, Features, ClassLabel

def MetaHatetest():
    return load_dataset("machlovi/MetaTest",split='Meta_test')
# Login using e.g. `huggingface-cli login` to access this dataset
# main_train_test = load_dataset("PKU-Alignment/PKU-SafeRLHF-V", "sexual_crimes")

def PolygloToxicityPrompts():
    dataset_wildchat = load_dataset("ToxicityPrompts/PolygloToxicityPrompts", "wildchat-en", split="wildchat")

    # Add a binary safety label
    def label_toxicity(example):
        example["prompt_label"] = "unsafe" if example["prompt_toxicity"] > 0.5 else "safe"
        return example

# Apply the transformation
    return  dataset_wildchat.map(label_toxicity)


def MoralExceptQA():
    ds = load_dataset("feradauto/MoralExceptQA", trust_remote_code=True)
    ds['test'] = ds['test'] \
        .rename_columns({
            "scenario": "prompt",
            "human.response": "prompt_label"
        }) \
        .map(lambda example: {
            **example,
            "prompt_label": "unsafe" if example["prompt_label"] > 0.5 else "safe"

        })
    return ds['test']


def load_beavertails():
    test = load_dataset("PKU-Alignment/BeaverTails", split='30k_test')
    def transform_labels(example):
        label = "safe" if example["is_safe"] else "unsafe"
        example["prompt_label"] = label
        example["response_label"] = label
        return example
    return test.map(transform_labels)




def load_toxic_chat():
    test = load_dataset("lmsys/toxic-chat", "toxicchat0124")
    for split in test:
        test[split] = test[split].rename_columns({
            "user_input": "prompt",
            "model_output": "response",
            "toxicity": "prompt_label"
        }).map(lambda x: {"prompt_label": "safe" if x["prompt_label"] == 0 else "unsafe"})
    return test['test']

dataset_loaders = {
    # "MetaHatetest":MetaHatetest,
    "PolygloToxicityPrompts":PolygloToxicityPrompts,
    "MoralExceptQA":MoralExceptQA,
    "BT":load_beavertails,
    "TOXC":load_toxic_chat
}


def data(dataset_name):
    if dataset_name in dataset_loaders:
        test = dataset_loaders[dataset_name]()
    else:
        raise ValueError(f"Dataset '{dataset_name}' not found!")
    print(test[0])
    return test
# %%
base_dir = "/home/naseem_fordham/LLM_evaluation/Moderators/Chatbase Moderators/Catplus_results"


import time
import csv
from openai import RateLimitError

def calculate_omni_moderation(dataset, csv_name):
    results = []

    # Resume from last checkpoint
    start_index = load_checkpoint()
    print(f"Resuming from checkpoint index: {start_index}")

    for i, example in enumerate(tqdm(dataset.select(range(start_index,len(dataset))), desc="Calculating calculate_omni_moderation"), start=start_index):
        # print(i)
        
        prompt = example.get("prompt", "")
        response = example.get("response", None)

        try:
            prompt_predic_label = check_moderator(prompt)
            prompt_result = prompt_predic_label.results[0]
            prompt_flagged = prompt_result.flagged

            response_flagged = ""
            response_text = ""
            response_label = ""

            if response is not None:
                response_predic_label = check_moderator(response)
                response_result = response_predic_label.results[0]
                response_flagged = response_result.flagged
                response_text = response
                response_label = example.get("response_label", "")

            row = [
                prompt,
                response_text,
                example.get("prompt_label", ""),
                response_label,
                example.get("source", ""),
                prompt_flagged,
                response_flagged
            ]
            results.append(row)

        except RateLimitError:
            save_checkpoint(i)
            print(f"Rate limit hit at index {i}. Sleeping for 60 minutes...")
            time.sleep(30)  # Sleep for 1 hour
            return
        except Exception as e:
            save_checkpoint(i)
            print(f"Error at index {i}: {e}. Saving checkpoint and exiting.")
            return

        if (i + 1) % 10 == 0:
            write_to_csv(csv_name, results)
            results.clear()
            save_checkpoint(i + 1)

    if results:
        write_to_csv(csv_name, results)
        save_checkpoint(len(dataset))


def write_to_csv(csv_name, rows):
    file_exists = os.path.exists(csv_name)

    with open(csv_name, mode="a", newline='') as file:
        writer = csv.writer(file)

        if not file_exists or os.path.getsize(csv_name) == 0:
            header = [
                "prompt", 
                "response", 
                "prompt_label", 
                "response_label", 
                "source",
                "prompt_flagged", 
                "response_flagged"
            ]
            writer.writerow(header)

        writer.writerows(rows)

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk, Sequence, Value, Features, ClassLabel

# final_dataset = load_dataset("machlovi/MetaTest",split='Meta_test')
# data_dict=final_dataset
# data_dict
csv_name="/home/naseem_fordham/LLM_evaluation/Moderators/Chatbase Moderators/Catplus_results/OAI_TOXC.csv"
# data_dict = load_from_disk(f"../dataX/New_DATA/combined_dataset_12")
# test_data = generate_text(data_dict)

test_data=data("TOXC")
# test

# test_data = data_dict
print(test_data)
calculate_omni_moderation(test_data, csv_name)



# def calculate_omni_moderation(dataset, csv_name):
#     results = []  # Store all results to write after processing

#     start_index = 0

#     for i, example in enumerate(tqdm(dataset[start_index:], desc="Calculating calculate_omni_moderation"), start=start_index):

#         prompt = example.get("prompt", "")
#         response = example.get("response", None)  # Explicitly check for None


#         try:
        
        
#             # Get moderation results for prompt
#             prompt_predic_label = check_moderator(prompt)
#             prompt_result = prompt_predic_label.results[0]
#             prompt_flagged = prompt_result.flagged

#             # Default values for response in case it's None
#             response_flagged = ""
#             response_text = ""
#             response_label = ""

#             if response is not None:
#                 response_predic_label = check_moderator(response)
#                 response_result = response_predic_label.results[0]
#                 response_flagged = response_result.flagged
#                 response_text = response
#                 response_label = example.get("response_label", "")

#             # Prepare data for CSV
#             row = [
#                 prompt,
#                 response_text,
#                 example.get("prompt_label", ""),
#                 response_label,
#                 example.get("source", ""),
#                 prompt_flagged,
#                 response_flagged
#             ]
#         except :
#                 save_checkpoint(i)
#                 print(f"Rate limit hit at index {i}. Saving checkpoint and exiting.")
#                 return

#         results.append(row)

#         if (i + 1) % 10 == 0:
#             with open(csv_name, mode="a", newline='') as file:
#                 writer = csv.writer(file)
                
#                 # Write the header if the file is empty
#                 if file.tell() == 0:
#                     header = [
#                     "prompt", 
#                     "response", 
#                     "prompt_label", 
#                     "response_label", 
#                     "source",
#                     "prompt_flagged", 
#                     "response_flagged"
#                 ]

#                     writer.writerow(header)
#                 writer.writerows(results)

#                 # Write the data rows
#                 writer.writerows(results)
#                 results = []  # Clear the results list

#             save_checkpoint(i + 1)

#     if results:
#         with open(csv_name, mode="a", newline='') as file:
#             writer = csv.writer(file)
            
#             # Write the header if the file is empty
#             if file.tell() == 0:
#                 header = [
#                 "prompt", 
#                 "response", 
#                 "prompt_label", 
#                 "response_label", 
#                 "source",
#                 "prompt_flagged", 
#                 "response_flagged"
#             ]

#                 writer.writerow(header)
#             writer.writerows(results)

#             # Write the data rows
#             writer.writerows(results)
#             results = []  # Clear the results list

#         save_checkpoint(len(dataset))



# calculate_omni_moderation(test_data, f"/home/naseem_fordham/LLM_evaluation/Moderators_results/fewshot_testing/OAI_Metahate_test.csv")

# %%
