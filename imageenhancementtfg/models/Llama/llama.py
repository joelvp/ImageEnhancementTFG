from typing import Optional

from models.Llama.objects.llama_response import LlamaResponse
from models.Llama.utils import *
from models.utils import load_config


class Llama:
    """
        A class to interact with the Llama API for generating responses based on text and image inputs.

        Attributes
        ----------
        config : dict
            Configuration loaded from 'data/config.ini'.
        llama_api : LlamaAPI
            Instance of the LlamaAPI.
        prompts : dict
            Dictionary containing the prompts.

        Methods
        -------
        generate(message: Optional[str], image: Optional[bool]) -> LlamaResponse:
            Generate a response based on the message and image provided.
        handle_image_and_text(message: str) -> LlamaResponse:
            Handle the case where both image and text are provided.
        handle_text(message: str) -> LlamaResponse:
            Handle the case where only text is provided.
        handle_image_only() -> LlamaResponse:
            Handle the case where only an image is provided.
        fetch_response_content(api_json: dict) -> LlamaResponse:
            Fetch the response content from the API.
        process_image_and_text_response(response_dict: dict, message: str) -> LlamaResponse:
            Process the response for image and text input.
        response_to_no_tasks(message: str) -> LlamaResponse:
            Generate a response when no tasks are detected.
        handle_error(error: Exception) -> LlamaResponse:
            Handle any errors that occur during processing.
        """
    def __init__(self):
        """
       Constructs all the necessary attributes for the Llama object.
       """
        self.config = load_config('data/config.ini')
        self.llama_api = LlamaAPI(self.config['api_keys']['llama'])
        self.prompts = load_json(self.config['llama']['prompts'])
        logging.info("Llama model initialized")

    def generate(self, message: Optional[str], image: Optional[bool]) -> LlamaResponse:
        """
        Generate a response based on the message and image provided.

        Parameters
        ----------
        message : Optional[str]
            The message content.
        image : Optional[bool]
            Flag indicating if an image is included.

        Returns
        -------
        LlamaResponse
            The generated response.
        """
        if image and message:
            logging.info("Both image and text provided. Processing image and text input...")
            return self.handle_image_and_text(message)
        elif not image and message:
            logging.info("Only text provided. Processing text input...")
            return self.handle_text(message)
        elif image and not message:
            logging.info("Only image provided. Processing image input...")
            return self.handle_image_only()

    def handle_image_and_text(self, message: str) -> LlamaResponse:
        """
        Handle the case where both image and text are provided.

        Parameters
        ----------
        message : str
            The message content.

        Returns
        -------
        LlamaResponse
            The response for image and text input.
        """
        prompt = get_prompt(self.prompts, 'task_chooser')
        api_json = create_api_json(prompt, f"Extraiga la informacion deseada de la siguiente frase:\n\n{message}")

        try:
            response_dict = get_response_dict(self.llama_api, api_json)
            return self.process_image_and_text_response(response_dict, message)
        except (RetryError, Exception) as e:
            return self.handle_error(e)

    def handle_text(self, message: str) -> LlamaResponse:
        """
        Handle the case where only text is provided.

        Parameters
        ----------
        message : str
            The message content.

        Returns
        -------
        LlamaResponse
            The response for text input.
        """
        prompt = get_prompt(self.prompts, 'plain_text')
        api_json = create_api_json(prompt, message)
        return self.fetch_response_content(api_json)

    def handle_image_only(self) -> LlamaResponse:
        """
        Handle the case where only an image is provided.

        Returns
        -------
        LlamaResponse
            The response for image input.
        """
        api_json = get_prompt(self.prompts, 'only_picture')
        return self.fetch_response_content(api_json)

    def fetch_response_content(self, api_json: dict) -> LlamaResponse:
        """
        Fetch the response content from the API.

        Parameters
        ----------
        api_json : dict
            The API JSON structure.

        Returns
        -------
        LlamaResponse
            The response content.
        """
        try:
            response = call_llama_api(self.llama_api, api_json)
            content = response['choices'][0]['message']['content']
            return LlamaResponse(response_text=content)
        except (RetryError, Exception) as e:
            return self.handle_error(e)

    def process_image_and_text_response(self, response_dict: dict, message: str) -> LlamaResponse:
        """
        Process the response for image and text input.

        Parameters
        ----------
        response_dict : dict
            The response dictionary from the API.
        message : str
            The message content.

        Returns
        -------
        LlamaResponse
            The processed response.
        """
        enhance = convert_str_to_bool(response_dict, 'mejora_imagen')
        sky_replacement = convert_str_to_bool(response_dict, 'cambio_cielo')

        if enhance and not sky_replacement:
            return LlamaResponse(tasks=response_dict['tareas'])
        elif sky_replacement:
            if "cambiando el cielo" not in response_dict['tareas']:
                response_dict['tareas'].append("cambiando el cielo")
            return LlamaResponse(tasks=response_dict['tareas'], new_background_image=response_dict['new_background_image'])
        else:
            return self.response_to_no_tasks(message)

    def response_to_no_tasks(self, message: str) -> LlamaResponse:
        """
        Generate a response when no tasks are detected.

        Parameters
        ----------
        message : str
            The message content.

        Returns
        -------
        LlamaResponse
            The response indicating no tasks.
        """
        prompt = get_prompt(self.prompts, 'no_tasks')
        api_json = create_api_json(prompt, message)
        try:
            response = call_llama_api(self.llama_api, api_json)
            content = response['choices'][0]['message']['content']
            return LlamaResponse(tasks=["No Tasks"], response_text=content)
        except (RetryError, Exception) as e:
            return self.handle_error(e)

    @staticmethod
    def handle_error(error: Exception) -> LlamaResponse:
        """
        Handle any errors that occur during processing.

        Parameters
        ----------
        error : Exception
            The exception that occurred.

        Returns
        -------
        LlamaResponse
            The error response.
        """
        logging.error(f"An error occurred: {error}")
        return LlamaResponse(error_message="Estoy teniendo fallos internos, dame otra oportunidad...")