import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
from dotenv import load_dotenv

load_dotenv()

def append_to_sheets(result):
    """
    data_dict: {'timestamp': ..., 'provider': ..., 'tps': ..., 'is_correct': ...}
    """
    try:
        # 1. 인증 설정
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            os.getenv("GOOGLE_SHEETS_JSON_PATH"), scope
        )
        client = gspread.authorize(creds)

        # 2. 시트 열기
        sheet = client.open(os.getenv("SPREADSHEET_NAME")).sheet1

        # 3. 데이터 행 추가 (리스트 형태로 변환)
        row = [
        result.get("timestamp"),
        result.get("provider"),
        result.get("model_requested"),
        round(result.get("tps", 0), 2),
        round(result.get("ttft", 0), 3),
        result.get("is_correct"),
        result.get("difficulty"),       # 이 부분이 잘 들어가는지 확인!
        result.get("subject"),
        result.get("system_fingerprint") # 이 부분이 잘 들어가는지 확인!
    ]
        sheet.append_row(row)
        print("Google Sheets 전송 완료!")
        
    except Exception as e:
        print(f"Google Sheets 전송 실패: {e}")

# 기존 run_experiment 함수 끝단에 append_to_sheets(result)를 추가하면 됩니다.