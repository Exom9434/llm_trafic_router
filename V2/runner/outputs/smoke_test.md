# 프로바이더 실지원 확인 결과

| 모델 | 상태 | 반환 모델 | logprob | fingerprint | 답 | 지연(ms) | 출력토큰 | 추론토큰 |
|---|---|---|:--:|---|:--:|---:|---:|---:|
| openai_gpt56_luna | OK | gpt-5.6-luna | X | — | C | 2813 | 4 | 0 |
| google_gemini_flash_lite | OK | gemini-3.5-flash-lite | X | — | C | 998 | 1 | — |
| deepseek_v4_flash | OK | deepseek-v4-flash | X | a26a7955944dc5c60445bff77fac9c8e | C | 661 | 23 | 21 |
| qwen_flash | OK | qwen3.7-flash-2026-07-15 | 거절 | — | C | 2436 | 230 | 224 |
| upstage_solar_pro3 | OK | solar-pro3-260323 | X | — | C | 921 | 3 | 0 |
| anthropic_haiku | OK | claude-haiku-4-5-20251001 | X | — | C | 962 | 4 | — |
| naver_hcx_dash | OK | HCX-DASH-002 | X | — | C | 301 | 2 | — |
| openai_gpt56_sol | OK | gpt-5.6-sol | X | — | C | 3296 | 4 | 0 |
| anthropic_sonnet5 | OK | claude-sonnet-5 | X | — | C | 1263 | 3 | — |

- `logprob = 거절`은 logprobs 파라미터를 넣으면 에러가 나고 빼면 되는 경우다.
- 정답은 C(Paris)다. `답` 칸이 C가 아니면 프롬프트나 파서를 손봐야 한다.
- **추론토큰이 0이 아니면 thinking 차단이 안 먹은 것이다.** `config.py`의
  `extra_body`를 고치거나 `direct_max_tokens`를 올려야 한다. 그냥 두면
  본실험 비용이 추정치를 크게 넘고, 심하면 빈 응답이 온다.
- `답` 칸이 비어 있는데 추론토큰이 상한에 가깝다면 추론이 출력 상한을 다 먹은 것이다.
