# Llama 3.2 구현 프로젝트

이 저장소는 다양한 Llama 3.2 모델 구현을 포함하고 있습니다. 기본 구현부터 인터넷 검색 기능이 포함된 RAG (Retrieval-Augmented Generation) 시스템, 그리고 한국어와 영어를 모두 지원하는 다양한 크기의 모델 구현을 제공합니다.

## 프로젝트 개요

### 1. 기본 Llama 구현 (bllossom.py)
- 한국어 Llama 3.2 모델 구현 (Bllossom/llama-3.2-Korean-Bllossom-3B)
- 간단한 질의응답 기능
- torch와 transformers 라이브러리 사용

### 2. 인터넷 검색 가능한 RAG 구현 (bllossom_internet.py)
- 인터넷 검색 기능이 통합된 고급 구현
- 주요 기능:
  - DuckDuckGo 검색 통합
  - 웹 콘텐츠 추출 및 처리
  - 컨텍스트 기반 응답 생성
  - 디버깅과 모니터링을 위한 로깅 시스템

### 3. Llama 3.2 1B 구현 (Llama3.2_1b.py)
- 1B 파라미터 모델을 사용한 경량화 구현
- 주요 기능:
  - Hugging Face 통합
  - 조정 가능한 텍스트 생성 파라미터
  - 메모리 효율적인 설정

### 4. Llama 3.2 3B Instruct 구현 (llmam3.2-8b.py)
- 3B 명령어 튜닝 모델 사용
- 1B 구현과 유사한 기능에 더 큰 모델 용량

## 필수 요구사항

```python
torch
transformers
duckduckgo_search
requests
beautifulsoup4
huggingface_hub
```

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install torch transformers duckduckgo_search requests beautifulsoup4 huggingface_hub
```

2. Hugging Face 인증 설정:
- Hugging Face에서 API 토큰 발급
- 스크립트의 "your token" 부분을 실제 토큰으로 교체

## 사용 방법

### 기본 한국어 모델
```python
python bllossom.py
```

### 인터넷 검색 가능 RAG 모델
```python
python bllossom_internet.py
```

### Llama 3.2 1B 모델
```python
python Llama3.2_1b.py
```

### Llama 3.2 3B Instruct 모델
```python
python llmam3.2-8b.py
```

## 설정 옵션

### 텍스트 생성 매개변수
- `max_new_tokens`: 생성할 최대 토큰 수 (기본값: 512)
- `temperature`: 생성의 무작위성 제어 (기본값: 0.7)
- `top_p`: 핵 샘플링 매개변수 (기본값: 0.95)
- `top_k`: 상위-k 샘플링 매개변수 (기본값: 40)
- `repetition_penalty`: 반복 방지 페널티 (기본값: 1.2)

### RAG 검색 매개변수
- `max_results`: 검색할 최대 결과 수 (기본값: 5)
- 사용자 정의 가능한 User Agent 및 요청 헤더
- 커스터마이즈 가능한 콘텐츠 추출 규칙

## 주요 기능

### RAG 구현 특징
- 인터넷 검색 통합
- 웹 콘텐츠 추출
- 텍스트 정제 및 처리
- 오류 처리 및 로깅
- 컨텍스트 기반 응답 생성

### 모델 구현 특징
- 다양한 모델 크기 옵션 (1B, 3B)
- 한국어 및 영어 지원
- 메모리 효율적인 설정
- 커스터마이즈 가능한 생성 파라미터

## 문제 해결

1. 메모리 부족 오류
- `device_map="auto"` 설정 확인
- `low_cpu_mem_usage=True` 옵션 사용
- 배치 크기 조정

2. 토큰화 관련 문제
- pad_token 설정 확인
- 토크나이저 특수 토큰 설정 확인

## 주의사항

- API 토큰은 반드시 안전하게 관리하세요
- 대용량 모델 사용 시 충분한 GPU 메모리 확보 필요
- RAG 구현 시 인터넷 연결 상태 확인 필요