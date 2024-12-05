from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login

def setup_llama(api_token):
    # Hugging Face API 로그인
    login(token=api_token)
    
    # Llama 3.2 1B 모델 로드
    model_name = "meta-llama/Llama-3.2-1B"
    
    # 토크나이저 초기화
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=api_token,
        trust_remote_code=True
    )
    # pad 토큰 설정
    tokenizer.pad_token = tokenizer.eos_token
    
    # 모델 초기화
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=api_token,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    return model, tokenizer

def generate_text(prompt, model, tokenizer, 
                 max_new_tokens=512,  # 생성할 최대 토큰 수를 지정
                 temperature=0.7,
                 top_p=0.95,
                 top_k=40,
                 num_return_sequences=1,
                 do_sample=True,  # 다양한 응답을 위해 샘플링 활성화
                 repetition_penalty=1.2):  # 반복 방지
    # 입력 텍스트 토크나이징
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 텍스트 생성
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,  # max_length 대신 max_new_tokens 사용
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_return_sequences=num_return_sequences,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,  # 반복 방지 페널티 추가
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3  # n-gram 반복 방지
    )
    
    # 생성된 텍스트 디코딩
    generated_texts = [
        tokenizer.decode(output, skip_special_tokens=True)
        for output in outputs
    ]
    
    return generated_texts

# 사용 예시
if __name__ == "__main__":
    # Hugging Face API 토큰
    api_token = "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"
    
    # 모델과 토크나이저 설정
    model, tokenizer = setup_llama(api_token)
    
    # 테스트 프롬프트 (더 구체적인 프롬프트 작성)
    prompt = """다음은 한국의 문화에 대한 포괄적인 설명입니다. 전통 문화, 현대 문화, 
    그리고 한국만의 독특한 특징을 포함하여 설명해주세요:"""
    
    # 텍스트 생성
    generated_texts = generate_text(prompt, model, tokenizer)
    
    # 결과 출력
    for i, text in enumerate(generated_texts, 1):
        print(f"생성된 텍스트 {i}:\n{text}\n")