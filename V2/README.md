# V2 — 부하 시간대 품질 저하 연구

이 폴더는 프로젝트 업그레이드 논의와 산출물을 모은 곳이다. 앞으로 메인 작업은 여기서 진행한다.

## 무엇이 담겼나

이 폴더는 두 가지를 담는다. 하나는 새 실험의 설계 문서이고, 다른 하나는 그 설계에 이르기까지 수행한 분석이다.

- **`EXPERIMENT_DESIGN.md`** — 새 실험의 전체 설계서. 가장 먼저 읽을 문서다. 원래 질문에서 왜 지금 설계로 왔는지, 측정 구성, 리뷰어 방어, 실행 순서까지 담았다.
- **`UPGRADE_ROADMAP.md`** — 업그레이드 초기의 우선순위 로드맵(P0/P1/P2). 설계 전환 이전에 작성했으며, 배경 맥락으로 남겨 둔다.
- **`runner/`** — 새 실험의 러너 코드. 지금은 보정 패스까지 구현되어 있다. `runner/README.md`가 실행법을 담는다.
- **`analysis/`** — 이번 업그레이드 과정에서 만든 분석 스크립트와 결과.

## 핵심 방향 한 줄 요약

상용 LLM API가 부하가 몰리는 시간대에 조용히 답변 품질을 떨어뜨리는지를, 예민한 지표(추론 토큰·일관성·fingerprint)로 프로바이더별 종단 측정해 검증한다.

## analysis/ 안의 산출물

세 분석은 각각 목적이 다르다.

- **`07_reanalysis_slot_level.py`** — 1차 논문의 통계 오류(pseudoreplication)를 슬롯 단위로 바로잡은 재분석. 결과: `outputs/reanalysis_slot_level_report.md`.
- **`08_accuracy_parsing_check.py`** — gpt-4o-mini의 49% 정확도가 파싱 오류가 아니라 MMLU-Pro에서의 실제 성능임을 확인. 결과: `outputs/accuracy_parsing_check.md`.
- **`09_power_cost_analysis.py`** — 1차 검정력·비용 계산기. logprob 주력과 두 표본 비교를
  전제해 폐기했다. 기록용으로만 남긴다.
- **`10_power_cost_v2.py`** — 검정력·비용 재산정(2026-08-21). 짝비교 모형으로 다시 계산하고
  단가는 `runner/config.py`에서 읽는다. 결과: `outputs/power_cost_v2.md`.

> **주의:** `07`·`08` 스크립트는 원래 V1 분석 파이프라인(`../analysis/common.py`와 `../analysis/outputs/cleaned_results.csv`)에 의존한다. 이곳의 사본은 기록용이며, 다시 돌리려면 원본 위치(`analysis/`)에서 실행하는 편이 안전하다. `09`는 파라미터만으로 독립 실행된다.

## 다음 할 일

설계서 10절의 실행 순서를 따른다.

1. ~~보정 패스 러너 작성~~ — 완료. `runner/`에 있다.
2. 보정 패스 실행 → 고정 문항 은행 + 모델별 노이즈 바닥선 확보 ← **여기**
3. 본실험 러너 완성 (지연 프로브 + 슬롯 스케줄러)
4. 2주 본실험
5. 분석·집필

2단계에 들어가기 전에 할 일은 세 가지다. 새 API 키 세 개(DeepSeek·Upstage·
CLOVA Studio)를 `.env`에 넣고, `runner/smoke_test.py`로 프로바이더 실지원을
확인하고, `runner/calibrate.py --dry-run`으로 비용을 먼저 본다.

3단계에서 새로 짜야 하는 것은 지연 프로브(스트리밍 TTFT 측정)와 하루 6~8슬롯
스케줄러뿐이다. 나머지 배선은 `runner/core.py`가 이미 갖고 있다.

## 구현 전 확인할 미확정 항목

2026-08-21 라인업 재조사로 대부분 정리됐다(설계서 부록 B). 남은 것은 셋이다.

1. 추론 차단이 실제로 먹는지 — smoke test의 추론토큰이 0인지 확인
2. HyperCLOVA X의 KRW 단가 — 요금 페이지를 못 읽어 비용 추정에서 빠져 있다
3. 검정력·비용 재산정 — logprob 전제가 깨져 `analysis/09_power_cost_analysis.py`를
   새 지표(자기일관성·추론 토큰)로 다시 돌려야 한다

## V1과의 관계

1차 논문과 원본 분석 파이프라인은 상위 폴더에 그대로 있다(`../report/`, `../analysis/`, `../experiment_api.py`). V2는 이들을 대체하지 않고, 새 실험을 위한 별도 작업 공간이다.
