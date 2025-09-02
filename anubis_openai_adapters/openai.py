"""ADAPTADOR PARA REALIZAR OPERACIONES CON LOS MODELOS LLM DE OPEN IA"""

import datetime
from anubis_core.models.ai_manager import AIRecipe
from anubis_core.ports.ai_service_manager import IAIServicesManagerPort

import json
import requests
class OpenAIAdapter(IAIServicesManagerPort):
    def __init__(self, token: str, endpoint: str = "https://api.openai.com/v1/chat/completions"):
        """Inicializador de clase. 

        Args:
            token (str): Tocken de acceso a la api de openai
        """
        
        self.token = token
        self.endpoint = endpoint
        """Token de acceso a la api de openai
        """

        self.cost_model_data = {
            "gpt-3.5-turbo" : [0.50, 1.50],
            "gpt-4o" : [2.5, 10],
        }

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}"
        }

    
    def _generate_ia_cost(self,model,messages,response,time_delta) -> AIRecipe:

        output=AIRecipe(
            id= response["id"],
            endpoint=self.endpoint,
            model=model,
            messages=messages["messages"],
            raw_response=response,
            tokens_in=response["usage"]["prompt_tokens"],
            tokens_out=response["usage"]["completion_tokens"],
            price_tokens_in=self.cost_model_data[model][0],
            price_tokens_out=self.cost_model_data[model][1],
            time_delta = str(time_delta)
        ) 
        
        return output
    
    
    
    def get_chat_completion(self, model, prompt,image_base64 = None, context=None) -> tuple[str, AIRecipe]:
        datetime_init = datetime.datetime.now()
        payload = {
                        "model": model,
                        "messages": [],
                        "max_tokens": 2000
                    }
         
        if context:
            payload["messages"].append({
                    "role": "system",
                    "content": context
                })
        
        content_user =  {
                    "role": "user",
                    "content": []
                }
        
        content_user["content"].append( {"type": "text", "text": prompt})

        if image_base64:
            content_user["content"].append( 
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }) 

        
        payload["messages"].append( content_user)

        ia_response = requests.post(self.endpoint, headers=self.headers, json=payload).json()
        time_delta =  datetime.datetime.now() - datetime_init

        if "choices" in ia_response.keys():
            response = ia_response["choices"][0]["message"]["content"]
            cost = self._generate_ia_cost(model,payload,ia_response,time_delta)
        else:
            print(ia_response)
            raise Exception("La ia no devuelve choices")

        return response , cost

    
    def get_chat_completion_json(self, model, prompt,image_base64 = None, context=None) -> tuple[dict, AIRecipe]:
        content_ia, cost = self.get_chat_completion(model,prompt,image_base64,context)
        contenido_str = content_ia.strip("```json").strip("```").strip()     
        return json.loads(contenido_str) , cost     
    