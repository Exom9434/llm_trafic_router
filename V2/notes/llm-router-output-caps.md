---
name: llm-router-output-caps
description: "V2 1차 보정 패스가 출력 상한 때문에 무효가 된 경위와 실측 결과 — 상한·파서·프롬프트를 만질 때 읽는다"
type: project
---

## 1차 보정 패스(2026-08-25 밤)는 폐기한다

33,120콜 중 32,836 성공, 오류 284건(429·타임아웃, 무해). **문제는 파싱 실패 5,697건(전체의 17%)이고 그중 5,659건이 출력 상한 소진이었다.**

| 모델 | 파싱실패 | 비율 | 당시 상한 |
|---|---:|---:|---:|
| deepseek_v4_flash | 3,077 | **71%** | 128 |
| upstage_solar_pro3 | 1,379 | 32% | 16 |
| anthropic_haiku | 1,038 | 24% | 8 |
| anthropic_sonnet5 | 185 | 13% | 8 |

**원인: 상한을 스모크 테스트("파리는 어디의 수도인가", 답 3~4토큰)로 정했다.** MMLU-Pro 40~85% 밴드 문항은 전혀 다르다. 스모크는 배선을 확인하지 토큰 수요를 확인하지 않는다. solar-pro4를 pro3로 되돌린 사고와 같은 종류인데, 그때 라인업 절반에 같은 지뢰가 남아 있는 걸 못 봤다.

**데이터를 못 쓰는 이유:** `select_bank.py`는 파싱 실패를 오답으로 세지 않고 제외한다(`correct=None`). 그런데 빠진 관측이 무작위가 아니다. 실패가 math·engineering 문항에 몰렸고(DeepSeek 588문항 중 547개가 2회 이상 실패), 한 문항 안에서 살아남은 시행은 토큰을 적게 쓴 것들 — 즉 모델이 쉽게 푼 경우다. 난이도가 실제보다 쉽게 추정되고 밴드가 잘못 잘린다. DeepSeek은 반복 6회 중 평균 2회만 남아 자기일관성 추정 자체가 성립하지 않는다.

## probe_caps 실측 (2026-08-26, 36문항 × 9모델, 상한 512 / 추론모델 16,384)

| 모델 | 파싱실패 | 그중 잘림 | 최대출력 |
|---|---:|---:|---:|
| upstage_solar_pro3 | 19 | 19 | 512 |
| anthropic_haiku | 15 | 10 | 512 |
| anthropic_sonnet5 | 6 | 2 | 512 |
| deepseek_v4_flash | 4 | 4 | **16,384** |
| qwen_flash | 0 | 0 | 6,282 |
| gemini / luna / sol / hcx | 0 | 0 | 1 / 4 / 4 / 12 |

## 원인이 둘로 갈렸다 (raw_text로 확인)

**haiku·sonnet5·solar-pro3 — 보이는 풀이다, 숨은 추론이 아니다.** raw_text가 "I need to find the net charge on a conducting sphere... Given information: ..." 식의 평문 CoT다. "한 글자만 답하라"는 지시를 어긴다. 어려운 문항에서만 그런다.

**DeepSeek — 진짜 숨은 추론.** raw_text가 **빈 문자열**인데 output_tokens가 16,384다. 16k로도 모자란다. `reasoning_effort`로 줄일 수 없다 — 추론 토큰이 DeepSeek의 주력 지표라 억누르면 재려는 것을 없애는 꼴이다.

**Qwen은 max_tokens가 안 먹는다.** 상한 768인데 1차에서 10,730, probe에서 6,282토큰. `enable_thinking=true` 경로에서 파라미터가 무시되는 것으로 보인다. 역설적으로 그래서 파싱 실패가 0이다.

## 파서 수정 (커밋 18fabe5)

풀이를 늘어놓고 **끝에 글자만 툭 놓는** 응답이 많았다. sonnet-5의 `... E = 2ix - 2iy + iz  D`, `The integral evaluates to pi*123 = 386.4158898 B`. 답이 분명히 있는데 기존 규칙 넷이 전부 놓쳤다.

`_TRAILING_LETTER`를 추가했다. **끝에서만, 앞이 글자·숫자가 아닐 때만** 인정한다 — 본문 중간 대문자를 주우면 조용한 오답이 된다. **잘린 응답에는 쓰지 않는다**(끊긴 자리의 글자는 답이 아니다). `core.py`가 output_tokens와 max_tokens를 비교해 `truncated`를 파서에 넘긴다. selftest 3건 추가.

## 설계에 미치는 영향

**"직답 프로브로 출력 8토큰에 묶는다"는 전제가 9개 중 4개에서 깨졌다.** 설계서 3.4절·4.2절을 손봐야 한다.

**역으로 주력 지표가 두꺼워질 수 있다.** 프롬프트가 "한 글자만"인데 수백 토큰을 쓴다면 그 대부분은 답이 아니라 추론이다. 설계서는 추론량을 DeepSeek·Qwen 둘에서만 잰다고 했는데, haiku·sonnet5·solar는 **보이는 풀이 길이(output_tokens)**로 같은 것을 잴 수 있다. 2개 → 5개.
**따라서 프롬프트를 더 강하게 눌러 CoT를 억누르면 안 된다.** 누르지 말고 재는 쪽이 맞다.

**검토 중인 아이디어 — 절단율을 지표로.** DeepSeek 상한을 감당 가능한 값에 고정하고 "상한을 넘는 비율"을 부하 지표로 쓴다. 부하로 추론이 길어지면 절단율이 오른다. 이진 결과라 싸고 비용이 유계다. 대가는 절단된 콜의 정확도·자기일관성 상실.

## 미해결

- **p50/p90/p99를 아직 못 봤다.** `uv run probe_caps.py --report-only`로 뽑아 상한을 정한다.
- **본실험 $85 추정이 무효.** `measured_output_tokens`(DeepSeek 72, haiku 4)가 같은 파리 문항에서 나온 값이다. DeepSeek이 평균 2,000토큰이면 그 모델만 $148로, 전체 예산이 무너진다.
- 재보정 비용은 1차 $7.42보다 크게 늘어난다.

관련: [[llm-router-redesign]] [[llm-router-power-cost]] [[llm-router-scheduling]]
