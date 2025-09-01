import datetime
import json
from anubis_core.ports.ai_service_manager import IAIServicesManagerPort
from anubis_core.models.ai_manager import AIRecipe

"""pip install google-genai    """
from google import genai


class GoogleGenAIAdapter(IAIServicesManagerPort):
    def __init__(self, token: str, endpoint: str = "https://ai.google.dev"):

        self.endpoint = endpoint

        self.client = genai.Client(api_key=token)

        self.cost_model_data = {
            "gemini-2.0-flash-lite" : [0.075, 0.30],           
            "gemini-2.5-flash" : [0.075, 0.30],           
            "gemini-2.5-pro-exp-03-25" : [0.15,3.50],            
        }

    def _generate_ia_cost(self,model,messages,response,time_delta) -> AIRecipe:
        return AIRecipe(
            id="SINID",
            endpoint=self.endpoint,
            model=model,
            messages=[messages],  
            raw_response={"value": response.text},
            tokens_in=response.usage_metadata.prompt_token_count,
            tokens_out=response.usage_metadata.candidates_token_count,
            price_tokens_in=self.cost_model_data[model][0],
            price_tokens_out=self.cost_model_data[model][1],
            time_delta = str(time_delta)
        ) 

    def get_chat_completion(self, model, prompt,image_base64 = None, context=None) -> tuple[str, AIRecipe]:
        datetime_init = datetime.datetime.now()

        contents = {
                "parts": [
                    
                ]
            }


        if context:
            contents["parts"].append({"text": context})


        if image_base64:
            contents["parts"].append( {
                        "inline_data": {
                            "mime_type": "image/jpeg",  # Ajusta según el tipo (png, webp, etc.)
                            "data": image_base64,
                        }
                    })
            
        contents["parts"].append({"text": prompt})

        response = self.client.models.generate_content(
                model=model,
                contents=contents,
            )
              

        time_delta =  datetime.datetime.now() - datetime_init
        
        cost = self._generate_ia_cost(model,contents,response,time_delta)

        return response.text , cost

    def get_chat_completion_json(self, model, prompt,image_base64 = None, context=None) -> tuple[dict, AIRecipe]:
        content_ia, cost = self.get_chat_completion(model,prompt,image_base64,context)
        
        contenido_str = content_ia.strip("```json").strip("```").strip()     
        contenido_str=  contenido_str.replace("```","")        
        contenido_str=  contenido_str.replace("'","")
        return json.loads(contenido_str) , cost        


