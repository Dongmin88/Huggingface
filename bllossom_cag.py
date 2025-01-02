import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CAGModel:
    def __init__(self, model_id: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.context = None
        
    def preload_knowledge(self, documents: list[str]):
        self.context = "\n\n".join(documents)
        logger.info("Knowledge context loaded")
        
    def generate_response(self, query: str) -> str:
        if self.context is None:
            raise ValueError("Context not loaded. Call preload_knowledge first.")
            
        prompt = f"{self.context}\n\n질문: {query}\n답변:"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True
        ).to(self.model.device)
        
        terminators = [
            self.tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(
            inputs.input_ids,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.2,  # 반복 패널티
        )
        
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        )
        return response
        
    def reset_cache(self):
        self.context = None
        logger.info("Context reset")

if __name__ == "__main__":
    model_id = 'Bllossom/llama-3.2-Korean-Bllossom-3B'
    cag_model = CAGModel(model_id)
    
    documents = [
        "세종대왕은 조선 제4대 왕이다.",
        "훈민정음을 창제하였다.",
        "집현전을 설립하였다."
    ]
    
    cag_model.preload_knowledge(documents)
    
    query = "세종대왕의 업적은 무엇인가?"
    response = cag_model.generate_response(query)
    print(f"질문: {query}")
    print(f"답변: {response}")