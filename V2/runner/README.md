# V2 runner — 보정 패스 러너

설계서(`../EXPERIMENT_DESIGN.md`) 10절 실행 순서의 1~2단계를 담당하는 코드다.
후보 문항 풀을 만들고, 저부하 시간에 라인업 전체를 통과시켜, 본실험이 쓸
고정 문항 은행과 모델별 노이즈 바닥선을 뽑는다.

본실험 러너(3단계)는 아직 없다. 다만 이 코드가 그 뼈대다. `core.py`의
`CallSpec`·`execute_call`은 phase와 slot만 바꾸면 본실험에 그대로 쓰인다.

## 빠르게 확인하기

API 키 없이 파이프라인 전체가 도는지 먼저 본다.

```bash
cd V2/runner
python selftest.py
```

가짜 어댑터로 864콜을 흉내 내고, 프롬프트·파싱·재개·선별·바닥선 산출을
점검한다. 전부 통과해야 실제 API를 쓸 준비가 된 것이다.

## 실행 순서

```bash
pip install -r requirements.txt

# 0. 프로바이더 실지원 확인 (모델당 2콜, 사실상 무료)
python smoke_test.py

# 1. 후보 문항 풀 만들기 (MMLU-Pro, 과목당 80문항)
python itembank.py --per-subject 80

# 2. 비용부터 확인
python calibrate.py --dry-run

# 3. 보정 패스 실행 (저부하 시간에)
python calibrate.py --k 5

# 4. 문항 은행 + 노이즈 바닥선
python select_bank.py --lo 0.40 --hi 0.85 --per-subject 60

# 5. 실측 토큰으로 본실험 비용 투영 + 지출 상한 산출
python budget.py
```

키 설정 상태가 궁금하면 아무 때나 `python check_env.py`를 친다.

3번은 중단해도 된다. 같은 명령을 다시 치면 이미 끝난 콜은 건너뛴다.

## 파일 구성

| 파일 | 하는 일 |
|---|---|
| `config.py` | 모델 라인업·가격·프로브 파라미터. 설계서 4.2절을 코드로 옮긴 것 |
| `prompts.py` | 프롬프트 조립, 답 글자 파싱, logprob → 선택지 확률분포 변환 |
| `calllog.py` | 콜 단위 로깅 스키마(설계서 4.3절)와 JSONL 기록·재개 |
| `providers/` | 어댑터 3종: OpenAI 호환, Anthropic, HyperCLOVA X |
| `core.py` | 콜 하나 실행 → CallRecord 하나. 보정 패스와 본실험이 공유한다 |
| `itembank.py` | MMLU-Pro 후보 풀 생성. 과목 균형, 1차 key audit 불량 문항 제외 |
| `calibrate.py` | 보정 패스 러너. 재개 가능, `--dry-run`으로 비용 선확인 |
| `select_bank.py` | 40~85% 밴드 선별 + 모델별 노이즈 바닥선 |
| `smoke_test.py` | logprob·fingerprint 실지원 확인 → `capabilities.json` |
| `check_env.py` | API 키 설정 상태 점검 (키 값은 찍지 않는다) |
| `budget.py` | 실측 토큰 기반 비용 투영, 하루 단위 지출 가드, 날짜 완주 기록 |
| `selftest.py` | API 없이 도는 자체 점검 |

## 설계상 짚어 둘 점

**파싱은 추측하지 않는다.** `parse_letter`는 답이 명확한 꼴이 아니면 None을
돌려준다. "I cannot answer"를 선택지 I로 읽는 식의 오독이 정확도를 조용히
오염시키는 쪽이, 파싱 실패로 집계되는 쪽보다 훨씬 위험하기 때문이다.
파싱 실패율은 문항·모델 단위로 리포트에 나온다.

**logprob 지원 여부는 실측으로 정한다.** `config.py`의 `supports_logprobs`가
`"unknown"`이면 일단 요청해 보고, `smoke_test.py`가 결과를 `capabilities.json`에
확정한다. 이후 실행은 이 파일을 읽는다.

**nonce는 system 메시지에만 붙인다.** 캐시 히트를 막되 문항 본문은 건드리지
않아야 문항 간 비교가 깨지지 않는다(설계서 4.1절).

**재개 키는 성공한 콜만 센다.** 오류로 끝난 레코드는 로그에 남지만 done 집합에
들어가지 않으므로 다시 실행하면 재시도된다.

**지출 가드는 하루 단위로만 끊는다.** 하루 도중에 멈추면 그날은 앞쪽 슬롯만
남아 절단이 시간대와 상관된다. 편향이 되는 것이다. 그래서 하루를 시작하기
전에 그날 치 예산을 예약하고, 모자라면 그날을 아예 시작하지 않는다. 완주
여부는 `outputs/day_status.jsonl`에 남고 분석은 완주한 날만 쓴다(설계서 8.1절).
정지는 모델 단위라 한 모델이 멈춰도 나머지는 계속 돈다.

## .env에 추가로 필요한 키

V1에서 쓰던 `OPENAI_API_KEY`·`ANTHROPIC_API_KEY`·`GOOGLE_API_KEY`·
`DASHSCOPE_API_KEY` 외에 세 개가 더 필요하다.

```
DEEPSEEK_API_KEY=...
UPSTAGE_API_KEY=...
CLOVASTUDIO_API_KEY=...
```

키가 없는 모델은 `get_models()`가 자동으로 빼거나, `--models`로 직접 고를 수 있다.

## 실행 전 확인이 남은 것

라인업은 2026-08-21에 전면 재조사해 갱신했다(설계서 부록 B). 남은 것은 셋이다.

1. **추론 차단이 실제로 먹는지** — `smoke_test.py` 결과의 추론토큰 칸이 0인지
   본다. 0이 아니면 `config.py`의 `extra_body`를 고쳐야 한다. 그냥 두면 비용이
   추정치를 크게 넘고, 심하면 빈 응답이 온다. Gemini 3.x는 완전 차단이 안 되므로
   `direct_max_tokens`를 넉넉히(24) 잡아 뒀다.
2. **HyperCLOVA X 단가** — KRW 요금표를 못 읽어 `price_in`/`price_out`이 0이다.
   콘솔에서 확인해 채울 것. 그 전까지 비용 추정에서 이 모델이 빠진다.
3. **Solar Pro 4의 logprob** — 미지원으로 판단했으나 확정은 smoke test 몫이다.

## 보정 패스 비용 추정

과목당 80문항(총 480) × (temp0 1콜 + 반복 5콜) 기준이다.

| 구성 | 콜 수 | 추정 비용 |
|---|---:|---:|
| 라인업 7개만 | 20,160 | 약 $2.5 |
| 앵커 2개 포함 9개 | 25,920 | 약 $10.6 |

앵커가 비용의 3/4를 먹는다(gpt-5.6-sol이 $6). 앵커를 후보 풀 전체에 돌릴
필요는 없으므로, `--models`로 앵커를 빼고 한 번 돌린 뒤 확정된 은행에만
앵커를 태우는 쪽이 싸다. HyperCLOVA 단가가 빠져 있어 실제 총액은 이보다 크다.
