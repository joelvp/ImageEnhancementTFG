import logging
from typing import Dict

from llamaapi import LlamaAPI

import json
from copy import deepcopy
from tenacity import retry, stop_after_attempt


def load_json(path: str) -> dict:
    """
    Load a JSON file.

    Parameters:
        path (str): Path to the JSON file.

    Returns:
        dict: Loaded JSON data.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_api_json(prompt: dict, message: str) -> dict:
    """
    Create API JSON structure.

    Parameters:
        prompt (dict): Prompt template.
        message (str): Message to add.

    Returns:
        dict: API JSON structure.
    """
    api_json = deepcopy(prompt)
    new_message = {"role": "user", "content": message}
    api_json["messages"].append(new_message)
    return api_json


@retry(stop=stop_after_attempt(3))
def call_llama_api(llama_api: LlamaAPI, api_json: dict) -> dict:
    """
    Call the llama API with retries.

    Parameters:
        llama_api: llama API instance.
        api_json (dict): API JSON structure.

    Returns:
        dict: API response.
    """
    logging.info("Calling llama API")
    response = llama_api.run(api_json)
    return response.json()


def get_response_dict(llama_api: LlamaAPI, api_json: dict) -> dict:
    """
    Get response dictionary from llama API.

    Parameters:
        llama_api: llama API instance.
        api_json (dict): API JSON structure.

    Returns:
        dict: Parsed response dictionary.
    """
    response = call_llama_api(llama_api, api_json)
    response_dict = response['choices'][0]['message']['function_call']['arguments']
    if isinstance(response_dict, str):
        response_dict = json.loads(response_dict)
    return response_dict


def convert_str_to_bool(response_dict: dict, key: str) -> bool:
    """
    Convert string values to boolean in a dictionary.

    Parameters:
        response_dict (dict): Dictionary containing the response.
        key (str): Key whose value needs to be converted.

    Returns:
        bool: Converted boolean value.
    """
    value = response_dict.get(key)
    if isinstance(value, str):
        return value.lower() == 'true'
    elif isinstance(value, bool):
        return value
    else:
        return False


def get_prompt(prompts_dict: Dict, task: str) -> Dict:
    """
    Retrieve a deep copy of a specific prompt from the prompts dictionary.

    Parameters:
        prompts_dict (Dict): Dictionary containing the prompts.
        task (str): The task name to retrieve the corresponding prompt.

    Returns:
        Dict: A deep copy of the specified prompt.
    """
    return deepcopy(prompts_dict["prompts"][task])
