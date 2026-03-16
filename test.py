import os
import time
import uuid
import json
import random
import re
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from datasets import load_dataset
from dashboards_gspread import append_to_sheets

# 1. 초기 설정 및 API 클라이언트 로드
load_dotenv()

import openai
from anthropic import Anthropic
from google import genai
from google.genai import types

oa_client = openai.OpenAI()
ant_client = Anthropic()
client_google = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# 데이터셋 로드
print("데이터셋 로딩 중... (최초 실행 시 시간이 걸릴 수 있습니다)")
mmlu_pro = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
mmlu_easy = load_dataset("cais/mmlu", "all", split="test")

# 2. CoT 전용 시스템 프롬프트 설정
SYSTEM_PROMPT = """You are a logical reasoning assistant. For every question, you must think step-by-step. 
You must format your entire response using the following XML-style tags:
<Reason> [Write your step-by-step reasoning here] </Reason>
<Answer> [Write only the single letter of the correct option here, e.g., A] </Answer>
Do not include any other text outside these tags."""

def record_benchmark(provider, model_name, prompt):
    request_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/{provider}_{timestamp}_{request_id}.json"
    
    metrics = {
        "request_id": request_id,
        "timestamp": timestamp,
        "provider": provider,
        "model_requested": model_name,
        "ttft": 0,
        "generation_time": 0,
        "tps": 0,
        "full_content": "",
        "reasoning": "",
        "parsed_answer": "",
        "system_fingerprint": "N/A", # None 대신 N/A로 초기화하면 시트에서 확인이 쉽습니다.
        "raw_response": None
    }

    start_time = time.time()
    first_token_time = None

    try:
        # --- API 호출 ---
        if provider == "openai":
            response = oa_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                stream=True
            )
            chunks = []
            for chunk in response:
                if first_token_time is None: first_token_time = time.time()
                if chunk.choices and chunk.choices[0].delta.content:
                    metrics["full_content"] += chunk.choices[0].delta.content
                
                # fingerprint가 들어있는 청크를 찾으면 metrics에 고정
                if hasattr(chunk, 'system_fingerprint') and chunk.system_fingerprint:
                    metrics["system_fingerprint"] = chunk.system_fingerprint
                
                chunks.append(chunk.model_dump())
            metrics["raw_response"] = chunks

        elif provider == "anthropic":
            with ant_client.messages.stream(
                model=model_name,
                system=SYSTEM_PROMPT,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                for text in stream.text_stream:
                    if first_token_time is None: first_token_time = time.time()
                    metrics["full_content"] += text
                
                # 최종 메시지에서 메타데이터 추출
                final_msg = stream.get_final_message()
                metrics["raw_response"] = final_msg.model_dump()
                # Anthropic은 모델ID가 지문 역할을 합니다
                metrics["system_fingerprint"] = final_msg.model

        elif provider == "google":
            response = client_google.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=2000
                )
            )
            if first_token_time is None: first_token_time = time.time()
            metrics["full_content"] = response.text
            
            # 구글 전용 메타데이터 저장
            metrics["system_fingerprint"] = response.model_version
            metrics["raw_response"] = {
                "model_version": response.model_version, 
                "usage": str(response.usage_metadata)
            }
        # --- 지표 계산 및 파싱 ---
        end_time = time.time()
        metrics["ttft"] = first_token_time - start_time if first_token_time else 0
        metrics["generation_time"] = end_time - first_token_time if first_token_time else 0
        
        # XML 파싱 로직
        content = metrics["full_content"]
        reason_match = re.search(r'<Reason>(.*?)</Reason>', content, re.DOTALL)
        answer_match = re.search(r'<Answer>(.*?)</Answer>', content, re.DOTALL)
        
        metrics["reasoning"] = reason_match.group(1).strip() if reason_match else ""
        metrics["parsed_answer"] = answer_match.group(1).strip().upper() if answer_match else ""
        
        # TPS 계산 (글자수 기반 근사치)
        metrics["tps"] = (len(content) / 4) / metrics["generation_time"] if metrics["generation_time"] > 0 else 0

        # 로그 저장
        os.makedirs("logs", exist_ok=True)
        with open(log_filename, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        return metrics
    except Exception as e:
        print(f"Error with {provider}: {e}")
        return None

def test_single_cycle():
    print(f"=== 테스트 사이클 시작: {datetime.now()} ===")
    
    targets = [
        # OpenAI: 성능은 5.0보다 월등하고 5.2보다는 훨씬 저렴한 모델
        {"provider": "openai", "model": "gpt-5-nano-2025-08-07"}, 
        
        # Anthropic: 작년 말 출시 이후 '황금 밸런스'로 불리는 4.5 소네트
        {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"}, 
        
        # Google: 어제 3.1 라인업 공개와 함께 표준이 된 프로 모델
        {"provider": "google", "model": "gemini-3.1-flash-lite-preview"} 
    ]
    
    # 테스트용으로 각 난이도별 딱 1문제씩만 추출
    # MMLU-Pro (Hard) 1문제 추출
    raw_pro = mmlu_pro.select(random.sample(range(len(mmlu_pro)), 1))[0]
    # MMLU (Easy) 1문제 추출
    raw_easy = mmlu_easy.select(random.sample(range(len(mmlu_easy)), 1))[0]

    # 공통 규격으로 변환
    test_problems = [
        {
            "difficulty": "hard",
            "question": raw_pro["question"],
            "options": raw_pro["options"],
            "answer": str(raw_pro["answer"]).upper(),
            "subject": raw_pro.get("category", "unknown") # .get("category")로 수정
        },
        {
            "difficulty": "easy",
            "question": raw_easy["question"],
            "options": raw_easy["choices"],
            "answer": chr(65 + int(raw_easy["answer"])),
            "subject": raw_easy.get("subject", "unknown") # Easy는 subject 유지
        }
    ]
    
    all_results = []

    # 이제 통합된 test_problems 리스트를 순회합니다.
    for prob in test_problems:
        question = f"Question: {prob['question']}\nOptions: {prob['options']}\nFollow the system prompt format."
        
        for target in targets:
            print(f"[{target['provider'].upper()}] {prob['difficulty']} 문제 테스트 중...")
            result = record_benchmark(target['provider'], target['model'], question)
            
            if result:
                result["difficulty"] = prob["difficulty"]
                result["subject"] = prob["subject"]
                result["correct_answer"] = prob["answer"]
                
                # 정규화된 정답(A, B, C...)과 모델이 뱉은 정답(parsed_answer) 비교
                result["is_correct"] = 1 if result["parsed_answer"] == prob["answer"] else 0
                all_results.append(result)
                
                # 구글 시트 전송 테스트
                try:
                    append_to_sheets(result)
                    print(f"   > 시트 전송 완료 (정답: {result['is_correct']})")
                except:
                    print("   > 시트 전송 실패 (설정을 확인하세요)")
            
            time.sleep(1)

    print(f"=== 테스트 완료! 결과 {len(all_results)}건 저장됨 ===")

if __name__ == "__main__":
    test_single_cycle()