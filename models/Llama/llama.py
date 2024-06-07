from copy import deepcopy

from llamaapi import LlamaAPI
import json
import configparser

from tenacity import retry, stop_after_attempt, wait_fixed, RetryError


class Llama:
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('data\config.ini')
        self.llama = LlamaAPI(config['api_keys']['llama'])
        self.prompts = self.load_json('data\prompts.json')

    def generate(self, message, image: bool):
        print("IMAGE", image, "MESSAGE", message)
        if image and message:
            prompt = self.get_prompt(self.prompts, 'task_chooser')
            api_json = self.add_message_to_dict(prompt, message)
            try:
                response = self.try_run(api_json).json()
                response_dict = response['choices'][0]['message']['function_call']['arguments']

                print("RESPONSE", response_dict)

                if isinstance(response_dict, str):
                    response_dict = json.loads(response_dict)
                if isinstance(response_dict['enhance_required'], str):
                    response_dict['enhance_required'] = response_dict['enhance_required'].lower() == 'true'
                if isinstance(response_dict['sky_replacement'], str):
                    response_dict['sky_replacement'] = response_dict['sky_replacement'].lower() == 'true'
                if response_dict['enhance_required'] and not response_dict['sky_replacement']:
                    return response_dict['tasks']
                elif response_dict['enhance_required'] and response_dict['sky_replacement']:
                    return ["Sky", response_dict['sky_background']]
                else:
                    return []
            except RetryError as e:
                print(f"An error occurred after retries: {e}")
                return "Estoy teniendo fallos internos, dame otra oportunidad..."
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                return "Estoy teniendo fallos internos, dame otra oportunidad..."

        elif not image and message:
            prompt = self.get_prompt(self.prompts, 'plain_text')
            api_json = self.add_message_to_dict(prompt, message)

        elif image and not message:
            api_json = self.get_prompt(self.prompts, 'only_picture')

        else:
            api_json = self.get_prompt(self.prompts, 'nothing')

        try:
            response = self.try_run(api_json).json()
            return response['choices'][0]['message']['content']
        except RetryError as e:
            print(f"An error occurred after retries: {e}")
            return "Estoy teniendo fallos internos, dame otra oportunidad..."
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return "Estoy teniendo fallos internos, dame otra oportunidad..."

    def add_message_to_dict(self, api_dict, message):
        new_message = {"role": "user", "content": message}
        api_dict["messages"].append(new_message)
        return api_dict

    def load_json(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_prompt(self, prompts, task):
        return deepcopy(prompts["prompts"][task])

    @retry(stop=stop_after_attempt(3))
    def try_run(self, api_json):
        print("Llama API call...")
        response = self.llama.run(api_json)
        return response

