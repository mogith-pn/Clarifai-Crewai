import logging
from typing import Any

from crewai import LLM, Agent, Crew, Process, Task
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededException,
)

from clarifai.client.model import Model


class ClarifaiLLM(LLM):
        def __init__(self,
                    model_url: str,
                    pat: str = None,
                    max_tokens: int = 1000,
                    temperature = 1,
                    top_p = 1,
                    base_url: str = "https://api.clarifai.com",
                    compute_cluster_id: str = None,
                    nodepool_id: str = None,
                    deployment_id: str = None,
                    **kwargs: Any
                    ):
            
            self.model_url = model_url
            self.pat = pat
            self.max_tokens = max_tokens
            self.temperature = temperature
            self.top_p = top_p
            self.base_url = base_url
            self.kwargs = kwargs
            self.compute_cluster_id = compute_cluster_id
            self.nodepool_id = nodepool_id
            self.deployment_id = deployment_id
            self.model = Model(url=model_url, pat=pat, base_url=base_url, 
                               compute_cluster_id=compute_cluster_id, nodepool_id=nodepool_id,
                               deployment_id=deployment_id)
                        
            
            
        def call(self, messages, callbacks= None,
                 available_functions = None,) -> str:
            """
            Call the LLM with the given prompt and return the response.
            """
            args = {
                "prompt": "You are a helpful assistant.",
                "chat_history": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                **self.kwargs
            }
            logging.debug(f"Calling LLM with args: {args}")
            try:
                response = self.model.predict(**args)
                return response
            except Exception as e:
                logging.error(f"request failed: {str(e)}")

                raise
            
        def supports_function_calling(self) -> bool:
            """Check if the LLM supports function calling.
            
            Returns:
                True if the LLM supports function calling, False otherwise.
            """
            # Return True if your LLM supports function calling
            return False