import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Tuple
import re
from urllib.parse import urljoin, urlparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    def _is_valid_url(self, url: str) -> bool:
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
            
    def scrape_page(self, url: str) -> str:
        if not self._is_valid_url(url):
            return ""
            
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
                
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {str(e)}")
            return ""
            
    def search(self, urls: List[str], max_pages: int = 5) -> List[Dict]:
        results = []
        for url in urls[:max_pages]:
            text = self.scrape_page(url)
            if text:
                results.append({
                    "url": url,
                    "content": text
                })
        return results

class CAGModelWithScraping:
    def __init__(self, model_id: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.scraper = WebScraper()
        self.context = None
        self.max_context_tokens = 2048
        
    def _clean_text(self, text: str) -> str:
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'[\n\r\t]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
        
    def _truncate_context(self, text: str) -> str:
        tokens = self.tokenizer.encode(text)
        if len(tokens) > self.max_context_tokens:
            tokens = tokens[:self.max_context_tokens]
            return self.tokenizer.decode(tokens)
        return text
        
    def preload_knowledge(self, documents: List[str], urls: List[str] = None):
        base_context = "\n\n".join(documents)
        
        if urls:
            scraped_results = self.scraper.search(urls)
            web_context = []
            
            for result in scraped_results:
                cleaned_text = self._clean_text(result['content'])[:1000]  # Limit each result
                web_context.append(cleaned_text)
            
            combined_context = base_context + "\n\n웹 검색 결과:\n" + "\n\n".join(web_context)
        else:
            combined_context = base_context
            
        self.context = self._truncate_context(combined_context)
        context_tokens = len(self.tokenizer.encode(self.context))
        logger.info(f"Knowledge context loaded (tokens: {context_tokens})")
        
    def generate_response(self, query: str) -> Tuple[str, Dict]:
        if self.context is None:
            raise ValueError("Context not loaded. Call preload_knowledge first.")
            
        prompt = f"{self.context}\n\n질문: {query}\n답변:"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True
        ).to(self.model.device)
        
        input_tokens = inputs.input_ids.shape[-1]
        logger.info(f"Input tokens: {input_tokens}")
        
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
            repetition_penalty=1.2,
        )
        
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        )
        
        generated_tokens = outputs.shape[-1] - input_tokens
        
        token_info = {
            "input_tokens": input_tokens,
            "generated_tokens": generated_tokens,
            "total_tokens": input_tokens + generated_tokens
        }
        
        return response, token_info
        
    def reset_cache(self):
        self.context = None
        logger.info("Context reset")

if __name__ == "__main__":
    model_id = 'Bllossom/llama-3.2-Korean-Bllossom-3B'
    cag_model = CAGModelWithScraping(model_id)
    
    documents = [
        "세종대왕은 조선 제4대 왕이다.",
        "훈민정음을 창제하였다.",
        "집현전을 설립하였다."
    ]
    
    urls = [
        "https://ko.wikipedia.org/wiki/세종대왕",
        "https://www.museum.go.kr/site/main/content/세종대왕"
    ]
    
    cag_model.preload_knowledge(documents, urls)
    
    query = "세종대왕의 업적은 무엇인가?"
    response, token_info = cag_model.generate_response(query)
    
    print(f"질문: {query}")
    print(f"답변: {response}")
    print("\n토큰 정보:")
    print(f"입력 토큰 수: {token_info['input_tokens']}")
    print(f"생성 토큰 수: {token_info['generated_tokens']}")
    print(f"전체 토큰 수: {token_info['total_tokens']}")