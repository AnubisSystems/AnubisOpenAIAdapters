import datetime
import json
from openai import OpenAI
import requests
from anubis_core.ports.ai_service_manager import IAIServicesManagerPort
from anubis_core.models.ai_manager import AIRecipe

class DeepSeekAdapter(IAIServicesManagerPort):
    def __init__(self, token: str, endpoint: str = "https://api.deepseek.com"):
        """Inicializador de clase para Alibaba Cloud Model Studio.

        Args:
            api_key (str): API Key para autenticación
            endpoint (str): Endpoint del modelo en Model Studio
        """

        self.endpoint = endpoint

        # Initialize OpenAI client
        self.client = OpenAI(         
            api_key = token,
            base_url=endpoint
        )

        self.cost_model_data = {
            "deepseek-chat" : [0.21, 0.63],        
        }

    def _generate_ia_cost(self,model,messages,response,time_delta) -> AIRecipe:
        return AIRecipe(
            id= response["id"],
            endpoint=self.endpoint,
            model=model,
            messages=messages,
            raw_response=response,
            tokens_in=response["usage"]["prompt_tokens"],
            tokens_out=response["usage"]["completion_tokens"],
            price_tokens_in=self.cost_model_data[model][0],
            price_tokens_out=self.cost_model_data[model][1],
            time_delta = str(time_delta)
        ) 

    def get_chat_completion(self, model, prompt,image_base64 = None, context=None) -> tuple[str, AIRecipe]:
        datetime_init = datetime.datetime.now()
        payload = []

        if context:
            payload.append({
                    "role": "system",
                    "content": context
                })



        content_user =  {
                            "role": "user",
                            "content": []
                        }

        if image_base64:
            content_user["content"].append( {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"})
        content_user["content"].append( {"type": "text", "text": prompt})
        payload.append( content_user)
        
        completion = self.client.chat.completions.create(
                model=model,                 
                    messages=payload,
                    stream=False
                )
        

        response_ia = completion.to_dict()
        time_delta =  datetime.datetime.now() - datetime_init
        response = response_ia["choices"][0]["message"]["content"]
        cost = self._generate_ia_cost(model,payload,response_ia,time_delta)

        return response , cost

    def get_chat_completion_json(self, model, prompt,image_base64 = None, context=None) -> tuple[dict, AIRecipe]:
        content_ia, cost = self.get_chat_completion(model,prompt,image_base64,context)
        contenido_str = content_ia.strip("```json").strip("```").strip()     

        return json.loads(contenido_str) , cost        


