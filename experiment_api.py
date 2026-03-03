import os
import time
import uuid
import json
import random
import schedule
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from datasets import load_dataset
from dashboards_gspread import append_to_sheets

# 1. 초기 설정
load_dotenv()

# 데이터셋 로드 (글로벌 변수)
print("데이터셋 로드 중...")
mmlu_pro = load_dataset("TIGER-Lab/MMLU-Pro", split="test", trust_remote_code=True)
mmlu_easy = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)

# API 클라이언트 사전 초기화
import openai
from anthropic import Anthropic
import google.generativeai as genai

# 각 API 키가 .env에 잘 있는지 확인용
oa_client = openai.OpenAI()
ant_client = Anthropic()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

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
        "system_fingerprint": None,
        "raw_response": None
    }

    start_time = time.time()
    first_token_time = None
    # 시스템 프롬프트 정의
    SYSTEM_PROMPT = "You are a benchmark assistant. Answer exactly with the option letter only. No prose."


    try:
        if provider == "openai":
            response = oa_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT}, # 시스템 메시지 추가
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
            chunks = []
            for chunk in response:
                if first_token_time is None: first_token_time = time.time()
                if chunk.choices and chunk.choices[0].delta.content:
                    metrics["full_content"] += chunk.choices[0].delta.content
                if hasattr(chunk, 'system_fingerprint'):
                    metrics["system_fingerprint"] = chunk.system_fingerprint
                chunks.append(chunk.model_dump())
            metrics["raw_response"] = chunks

        elif provider == "anthropic":
            with ant_client.messages.stream(
                model=model_name,
                system=SYSTEM_PROMPT, # 여기에 위치
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            ) as stream:
                for text in stream.text_stream:
                    if first_token_time is None: first_token_time = time.time()
                    metrics["full_content"] += text
                metrics["raw_response"] = stream.get_final_message().model_dump()

        elif provider == "google":
            model = genai.GenerativeModel(model_name, system_instruction=SYSTEM_PROMPT)
            response = model.generate_content(prompt, stream=True)
            for chunk in response:
                if first_token_time is None: first_token_time = time.time()
                metrics["full_content"] += chunk.text
            metrics["raw_response"] = {"text": metrics["full_content"]}

        # 지표 계산
        end_time = time.time()
        metrics["ttft"] = first_token_time - start_time if first_token_time else 0
        metrics["generation_time"] = end_time - first_token_time if first_token_time else 0
        
        # 글자 수 기반 TPS 계산 (한글/영어 혼용 고려)
        token_count = len(metrics["full_content"]) / 4 
        metrics["tps"] = token_count / metrics["generation_time"] if metrics["generation_time"] > 0 else 0

        os.makedirs("logs", exist_ok=True)
        with open(log_filename, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        return metrics

    except Exception as e:
        print(f"Error with {provider}: {e}")
        return None

def run_experiment():
    targets = [
        {"provider": "openai", "model": "gpt-4o"},
        {"provider": "anthropic", "model": "claude-3-5-sonnet-20240620"},
        {"provider": "google", "model": "gemini-1.5-pro"}
    ]
    
    pro_probs, easy_probs = get_random_problems(n=3)
    all_results = []

    for category, probs in [("hard", pro_probs), ("easy", easy_probs)]:
        for prob in probs:
            question = f"Question: {prob['question']}\nOptions: {prob['options']}\nAnswer with only the option letter (A, B, C, D, or E)."
            
            for target in targets:
                result = record_benchmark(target['provider'], target['model'], question)
                
                if result:
                    result["difficulty"] = category
                    result["subject"] = prob.get("subject", "unknown")
                    result["correct_answer"] = prob["answer"]
                    
                    # 정답 체크 로직 개선: 공백 제거 후 첫 글자 비교 혹은 포함 여부
                    pred = result["full_content"].strip().upper()
                    result["is_correct"] = 1 if (len(pred) > 0 and pred[0] == prob["answer"]) else 0
                    
                    all_results.append(result)
                    
                    # 실시간 전송 (하나씩 전송하여 유실 방지)
                    append_to_sheets(result) 
                
                time.sleep(2) 

    # 로컬 백업용 CSV 저장
    df = pd.DataFrame(all_results)
    header = not os.path.exists("benchmark_results.csv")
    df.to_csv("benchmark_results.csv", mode='a', index=False, header=header)
    print(f"실험 사이클 완료: {datetime.now()}")


def get_random_problems(n=5):
    """실험마다 매번 다른 문제를 추출하여 캐싱 방지"""
    pro_samples = mmlu_pro.select(random.sample(range(len(mmlu_pro)), n))
    easy_samples = mmlu_easy.select(random.sample(range(len(mmlu_easy)), n))
    return list(pro_samples), list(easy_samples)

def job():
    print(f"[{datetime.now()}] 벤치마크 테스트 시작...")
    run_experiment() # 위에서 만든 실험 함수 호출

# 1. 논-피크 시간대 수집 (KST 오후 6시부터 15분 간격으로 5회 실행)
schedule.every().day.at("18:00").do(job)
schedule.every().day.at("18:15").do(job)
schedule.every().day.at("18:30").do(job)
schedule.every().day.at("18:45").do(job)
schedule.every().day.at("19:00").do(job)

# 2. 피크 시간대 수집 (KST 오전 6시부터 15분 간격으로 5회 실행)
schedule.every().day.at("06:00").do(job)
schedule.every().day.at("06:15").do(job)
schedule.every().day.at("06:30").do(job)
schedule.every().day.at("06:45").do(job)
schedule.every().day.at("07:00").do(job)

print("스케줄러 시작... 대기 중.")
while True:
    schedule.run_pending()
    time.sleep(60)