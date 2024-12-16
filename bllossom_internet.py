import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
from typing import List, Dict

class RAGSearchLLM:
    def __init__(self, model_id: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.ddgs = DDGS()
        
    def search_internet(self, query: str, max_results: int = 3) -> List[str]:
        """인터넷 검색을 수행하고 결과를 반환합니다."""
        results = self.ddgs.text(query, max_results=max_results)
        contents = []
        
        for result in results:
            try:
                response = requests.get(result['link'])
                soup = BeautifulSoup(response.text, 'html.parser')
                # 본문 내용 추출
                text = ' '.join([p.text for p in soup.find_all('p')])
                contents.append(text[:500])  # 각 결과의 처음 500자만 사용
            except:
                continue
                
        return contents

    def generate_response(self, instruction: str, search_results: List[str]) -> str:
        """검색 결과를 포함하여 응답을 생성합니다."""
        # 프롬프트 템플릿 구성
        context = "\n".join(search_results)
        prompt = f"""다음은 관련된 정보입니다:
{context}

질문: {instruction}
답변:"""
        
        messages = [{"role": "user", "content": prompt}]
        
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        terminators = [
            self.tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9
        )
        
        return self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

    def query(self, instruction: str) -> str:
        """검색과 응답 생성을 통합하여 실행합니다."""
        search_results = self.search_internet(instruction)
        return self.generate_response(instruction, search_results)

# 사용 예시
if __name__ == "__main__":
    model_id = 'Bllossom/llama-3.2-Korean-Bllossom-3B'
    rag_llm = RAGSearchLLM(model_id)
    
    response = rag_llm.query("뉴진스 가수에 대해 알려줘 2022년 데뷔했다. 대표적인 곡 알려줘")
    print(response)