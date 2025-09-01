from abc import ABC, abstractmethod
import datetime
from huggingface_hub import InferenceClient
from typing import Optional, Tuple, Dict
import base64
import io
from PIL import Image
import requests

from anubis_core.models.ai_manager import AIRecipe
from anubis_core.ports.ai_service_manager import IAIServicesManagerPort


"""
    pip install huggingface_hub  Pillow
"""    
class HuggingFaceAdapter(IAIServicesManagerPort):

    def __init__(self, token: str):
        self.token = token
        self.client = InferenceClient(token=token)

    def _decode_base64_image(self, base64_string: str) -> bytes:
        image_data = base64.b64decode(base64_string)
        return image_data
    

    def _generate_ia_cost(self,model,messages,response,time_delta) -> AIRecipe:
        return AIRecipe(
            id="SINID",
            endpoint=f"https://api-inference.huggingface.co/models/{model}",
            model=model,
            messages=[messages],  
            raw_response={"value": response},
            tokens_in=1,
            tokens_out=1,
            price_tokens_in=0.01,
            price_tokens_out=0.01,
            time_delta = str(time_delta)
        ) 

    def get_chat_completion(self, model, prompt, image_base64=None, context=None) -> Tuple[str, AIRecipe]:
        datetime_init = datetime.datetime.now()
        full_prompt = f"{context}\n{prompt}" if context else prompt
        if image_base64:
            image_bytes = self._decode_base64_image(image_base64)
            image = Image.open(io.BytesIO(image_bytes))

            # Aquí se llama al método multimodal
            response = self._call_huggingface_multimodal_api(model,full_prompt,image_base64)
            
        else:            
            response = self.client.text_generation(prompt=full_prompt, model=model)

        time_delta =  datetime.datetime.now() - datetime_init

        return response, self._generate_ia_cost(model,[full_prompt,image_base64],response,time_delta)

    def get_chat_completion_json(self, model, prompt, image_base64=None, context=None) -> Tuple[Dict, AIRecipe]:
        output_text, recipe = self.get_chat_completion(model, prompt, image_base64, context)
        response_json = {
            "model": model,
            "prompt": prompt,
            "response": output_text
        }
        return response_json, recipe
    
    def _call_huggingface_multimodal_api(self, model: str, prompt: str, image_base64: str) -> str:
        API_URL = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {self.token}"}

        image_bytes = base64.b64decode(image_base64)

        payload = {
            "inputs": {
                "image": image_bytes,
                "question": prompt
            }
        }

        response = requests.post(API_URL, headers=headers, files={"image": image_bytes}, data={"inputs": prompt})
        response.raise_for_status()

        result = response.json()

        # Dependiendo del modelo puede devolver un string o lista
        if isinstance(result, list) and 'generated_text' in result[0]:
            return result[0]['generated_text']
        elif isinstance(result, list) and 'answer' in result[0]:
            return result[0]['answer']
        elif isinstance(result, dict) and 'generated_text' in result:
            return result['generated_text']
        else:
            return str(result)