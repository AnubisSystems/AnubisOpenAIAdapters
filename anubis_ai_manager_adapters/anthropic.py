import datetime
import json
import anthropic
import requests
from anubis_core.ports.ai_service_manager import IAIServicesManagerPort
from anubis_core.models.ai_manager import AIRecipe

"""
TODO: NO ESTA PROBADO POR FALTA DE TOKENS. PARECE CARO A MIRAR SI MERECE
        https://www.anthropic.com/
"""

class AnthropicAdapter(IAIServicesManagerPort):
    def __init__(self, token: str, endpoint: str = "https://www.anthropic.com/"):

        self.endpoint = endpoint

        # Initialize OpenAI client
        self.client = anthropic.Anthropic(
                    # defaults to os.environ.get("ANTHROPIC_API_KEY")
                    api_key=token,
                )

        self.cost_model_data = {
            "claude-3-7-sonnet" : [3, 15],
            "claude-3-opus" : [15, 75],
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
        
        chat_message = self.client.messages.create(
                    model=model,
                    max_tokens=1024,
                    messages=payload
                )        

        time_delta =  datetime.datetime.now() - datetime_init
        response = chat_message.content
        cost = self._generate_ia_cost(model,payload,chat_message,time_delta)

        return response , cost

    def get_chat_completion_json(self, model, prompt,image_base64 = None, context=None) -> tuple[dict, AIRecipe]:
        content_ia, cost = self.get_chat_completion(model,prompt,image_base64,context)
        contenido_str = content_ia.strip("```json").strip("```").strip()     

        return json.loads(contenido_str) , cost        


