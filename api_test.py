import os
from dotenv import load_dotenv
import openai
from anthropic import Anthropic
from google import genai
from google.genai import types

# 1. 초기 설정
load_dotenv()

# 2. 로드된 직후 바로 출력을 해보세요. (보안을 위해 앞글자만)
qwen_key = os.getenv("DASHSCOPE_API_KEY")

if qwen_key is None:
    print("❌ [경고] .env 파일을 찾지 못했거나 변수가 없습니다.")
else:
    print(f"✅ [확인] 키 로드 성공: {qwen_key[:5]}***")
    
# 클라이언트 객체 생성
oa_client     = openai.OpenAI()
ant_client    = Anthropic()
google_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
qwen_client   = openai.OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1"
)

TEST_PROMPT = "What is 1+1? Answer with the number only."
#5 mini와 나노는 temp설정이 안돼어서 4o로 테스트합니다. (2026-03-18 기준)
MODELS = {
    "openai":    "gpt-4o-mini-2024-07-18",
    "anthropic": "claude-haiku-4-5-20251001",
    "google":    "gemini-3.1-flash-lite-preview",
    "qwen":      "qwen3.5-flash"  # Qwen 추가
}

# 2. 테스트 함수 정의
def test_openai():
    response = oa_client.chat.completions.create(
        model=MODELS["openai"],
        messages=[{"role": "user", "content": TEST_PROMPT}],
        temperature=0,
        stream=False # 테스트용이므로 간단히 False
    )
    return response.choices[0].message.content.strip()

def test_anthropic():
    response = ant_client.messages.create(
        model=MODELS["anthropic"],
        max_tokens=10,
        temperature=0,
        messages=[{"role": "user", "content": TEST_PROMPT}]
    )
    return response.content[0].text.strip()

def test_google():
    # 2026년 최신 SDK 방식 적용
    response = google_client.models.generate_content(
        model=MODELS["google"],
        contents=TEST_PROMPT,
        config=types.GenerateContentConfig(temperature=0, max_output_tokens=10)
    )
    return response.text.strip()

def test_qwen():
    # DashScope 호환 모드 적용
    response = qwen_client.chat.completions.create(
        model=MODELS["qwen"],
        messages=[{"role": "user", "content": TEST_PROMPT}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

TESTS = {
    "openai":    test_openai,
    "anthropic": test_anthropic,
    "google":    test_google,
    "qwen":      test_qwen,
}

# 3. 실행부
if __name__ == "__main__":
    print(f"🚀 2026 벤치마크 사전 테스트 시작 (KST: {os.popen('date').read().strip()})")
    print("-" * 60)
    
    for provider, fn in TESTS.items():
        model = MODELS[provider]
        try:
            answer = fn()
            print(f"✅ [SUCCESS] {provider:10} | Model: {model:30} | Ans: {answer}")
        except Exception as e:
            print(f"❌ [FAILED ] {provider:10} | Model: {model:30}")
            
            # 1. 에러 메시지 전체 출력 (repr 사용 시 타입까지 확인 가능)
            print(f"   Error Type: {type(e).__name__}")
            print(f"   Full Message: {e}") 
            
            # 2. OpenAI/Qwen 같은 API 에러일 경우 상세 바디 정보 출력
            if hasattr(e, 'body'):
                print(f"   API Details: {e.body}")
            
            print("-" * 30) # 구분선 추가

    print("-" * 60)
    print("테스트 종료.")