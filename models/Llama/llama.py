from copy import deepcopy
from llamaapi import LlamaAPI
import json
import configparser
from tenacity import retry, stop_after_attempt, RetryError

from models.Llama.objects.llama_response import LlamaResponse


class Llama:
    def __init__(self):
        self.config = self.load_config('data/config.ini')
        self.llama_api = LlamaAPI(self.config['api_keys']['llama'])
        self.prompts = self.load_json('data/prompts.json')

    def load_config(self, path):
        config = configparser.ConfigParser()
        config.read(path)
        return config

    def load_json(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def generate(self, message, image) -> LlamaResponse:
        if image and message:
            return self.handle_image_and_text(message)
        elif not image and message:
            return self.handle_text(message)
        elif image and not message:
            return self.handle_image_only()
        else:
            return self.handle_no_input()

    def handle_image_and_text(self, message) -> LlamaResponse:
        prompt = self.get_prompt('task_chooser')
        api_json = self.create_api_json(prompt, f"Extraiga la informacion deseada de la siguiente frase:\n\n{message}")

        try:
            response_dict = self.get_response_dict(api_json)
            return self.process_image_and_text_response(response_dict, message)
        except (RetryError, Exception) as e:
            return self.handle_error(e)

    def handle_text(self, message) -> LlamaResponse:
        prompt = self.get_prompt('plain_text')
        api_json = self.create_api_json(prompt, message)
        return self.fetch_response_content(api_json)

    def handle_image_only(self) -> LlamaResponse:
        api_json = self.get_prompt('only_picture')
        return self.fetch_response_content(api_json)

    def handle_no_input(self) -> LlamaResponse:
        api_json = self.get_prompt('nothing')
        return self.fetch_response_content(api_json)

    def create_api_json(self, prompt, message):
        api_json = deepcopy(prompt)
        new_message = {"role": "user", "content": message}
        api_json["messages"].append(new_message)
        return api_json

    @retry(stop=stop_after_attempt(3))
    def call_llama_api(self, api_json):
        print("Llama API call...")
        response = self.llama_api.run(api_json)
        return response

    def get_response_dict(self, api_json):
        response = self.call_llama_api(api_json).json()
        response_dict = response['choices'][0]['message']['function_call']['arguments']
        if isinstance(response_dict, str):
            response_dict = json.loads(response_dict)
        return response_dict

    def fetch_response_content(self, api_json) -> LlamaResponse:
        try:
            response = self.call_llama_api(api_json).json()
            content = response['choices'][0]['message']['content']
            return LlamaResponse(response_text=content)
        except (RetryError, Exception) as e:
            return self.handle_error(e)

    def process_image_and_text_response(self, response_dict, message) -> LlamaResponse:
        self.convert_str_to_bool(response_dict, 'mejora_imagen')
        self.convert_str_to_bool(response_dict, 'cambio_cielo')

        if response_dict['mejora_imagen'] and not response_dict['cambio_cielo']:
            return LlamaResponse(tasks=response_dict['tareas'])
        elif response_dict['cambio_cielo']:
            return LlamaResponse(tasks=["Sky"], new_background_image=response_dict['new_background_image'])
        else:
            return self.response_to_no_tasks(message)

    def convert_str_to_bool(self, response_dict, key):
        if isinstance(response_dict.get(key), str):
            response_dict[key] = response_dict[key].lower() == 'true'

    def response_to_no_tasks(self, message) -> LlamaResponse:
        prompt = self.get_prompt('no_tasks')
        api_json = self.create_api_json(prompt, message)
        try:
            response = self.call_llama_api(api_json).json()
            content = response['choices'][0]['message']['content']
            return LlamaResponse(tasks=["No Tasks"], response_text=content)
        except (RetryError, Exception) as e:
            return self.handle_error(e)

    def get_prompt(self, task):
        return deepcopy(self.prompts["prompts"][task])

    def handle_error(self, error) -> LlamaResponse:
        print(f"An error occurred: {error}")
        return LlamaResponse(error_message="Estoy teniendo fallos internos, dame otra oportunidad...")