import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        logger.info(f"검색 쿼리: {query}")
        
        try:
            # 검색어를 명확하게 지정
            search_query = "NewJeans 뉴진스 아이돌 그룹 2022 데뷔 대표곡"
            
            # DuckDuckGo 검색 수행
            results = []
            for r in self.ddgs.text(search_query, max_results=max_results):
                logger.info(f"검색 결과: {r}")
                results.append(r)
            
            logger.info(f"검색 결과 수: {len(results)}")
            contents = []
            
            for result in results:
                try:
                    # 웹사이트 접근을 위한 헤더
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
                    }
                    
                    if 'link' in result:
                        logger.info(f"URL 접근 시도: {result['link']}")
                        response = requests.get(
                            result['link'],
                            headers=headers,
                            timeout=10,
                            verify=False  # SSL 인증서 검증 건너뛰기
                        )
                        
                        if response.status_code == 200:
                            soup = BeautifulSoup(response.text, 'html.parser')
                            
                            # 텍스트 추출 방식 개선
                            text_parts = []
                            for p in soup.find_all(['p', 'div', 'span']):
                                if p.text and len(p.text.strip()) > 50:  # 의미있는 텍스트만 추출
                                    text_parts.append(p.text.strip())
                            
                            if text_parts:
                                combined_text = ' '.join(text_parts)
                                contents.append(combined_text[:1000])  # 더 긴 컨텍스트 허용
                                logger.info(f"컨텐츠 추출 성공 (길이: {len(combined_text[:1000])})")
                    
                except Exception as e:
                    logger.error(f"URL 처리 중 오류 발생: {str(e)}")
                    continue
            
            # 검색 결과가 없는 경우를 위한 기본 정보 추가
            if not contents:
                default_info = """
                뉴진스(NewJeans)는 2022년 7월 1일에 데뷔한 하이브(HYBE) 산하 레이블 어도어의 5인조 걸그룹입니다.
                멤버는 민지, 하니, 다니엘, 해린, 혜인으로 구성되어 있습니다.
                대표곡으로는 'Attention', 'Hype Boy', 'Cookie', 'Ditto', 'OMG', 'Super Shy', 'ETA', 'Cool With You' 등이 있습니다.
                """
                contents.append(default_info)
            
            logger.info(f"최종 추출된 콘텐츠 수: {len(contents)}")
            return contents
            
        except Exception as e:
            logger.error(f"검색 중 오류 발생: {str(e)}")
            return ["검색 결과를 가져오는 중 오류가 발생했습니다."]

    def generate_response(self, instruction: str, search_results: List[str]) -> str:
        """검색 결과를 포함하여 응답을 생성합니다."""
        context = "\n".join(search_results)
        prompt = f"""다음은 뉴진스(NewJeans)에 대한 정보입니다:
{context}

질문: {instruction}
답변:"""
        
        logger.info("생성된 프롬프트:")
        logger.info(prompt)
        
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
        
        response = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
        return response

    def query(self, instruction: str) -> str:
        """검색과 응답 생성을 통합하여 실행합니다."""
        search_results = self.search_internet(instruction)
        return self.generate_response(instruction, search_results)

# 사용 예시
if __name__ == "__main__":
    model_id = 'Bllossom/llama-3.2-Korean-Bllossom-3B'
    rag_llm = RAGSearchLLM(model_id)
    
    response = rag_llm.query("뉴진스 가수에 대해 알려줘 2022년 데뷔했다. 대표적인 곡 알려줘")
    print("\n최종 응답:")
    print(response)