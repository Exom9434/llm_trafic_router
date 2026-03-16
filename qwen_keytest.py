import os
from dotenv import load_dotenv

# 2. 로드된 직후 바로 출력을 해보세요. (보안을 위해 앞글자만)
load_dotenv()
qwen_key = os.getenv("DASHSCOPE_API_KEY")

if qwen_key is None:
    print("❌ [경고] .env 파일을 찾지 못했거나 변수가 없습니다.")
else:
    print(f"✅ [확인] 키 로드 성공: {qwen_key[:5]}***")