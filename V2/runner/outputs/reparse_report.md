# 재파싱 결과

- 로그: `calibration_calls.jsonl` (33,769행)
- 값이 바뀌는 콜: 510건 (1.51%)
- 시각: 2026-09-03T07:18:08+00:00

## 모델별

| 모델 | 답이 바뀜 | 읽던 답을 버림 | 못 읽던 답을 읽음 | 합계 |
|---|---:|---:|---:|---:|
| anthropic_haiku | 14 | 0 | 38 | 52 |
| upstage_solar_pro3 | 57 | 399 | 2 | 458 |

## 정답률 변화

10지선다이므로 우연 수준은 0.100이다. 버린 답의 정답률이 우연에 가까우면
그 값들은 답이 아니라 잡음이었다는 뜻이다.

| 변화 | 건수 | 이전 | 이후 |
|---|---:|---:|---:|
| 답이 바뀜 | 71 | 0.099 | 0.648 |
| 읽던 답을 버림 | 399 | 0.333 | 0.000 |
| 못 읽던 답을 읽음 | 40 | 0.000 | 0.750 |

## 예시 — 답이 바뀜

- `math:8866 / upstage_solar_pro3 / D -> F (정답 F) ::  Wait options: A 102, B 155, C 86, D 108, E 75, F 117, G 125, H 140, I 130, J 94. So F is 117. Thus answer: F.`
- `math:8746 / upstage_solar_pro3 / A -> B (정답 B) ::  + \frac{1}{3}$ $2 = 1 + 0.333 = 1.333$, so that's also false. So our answer is definitely B. 1.414. Answer: B`
- `math:8015 / upstage_solar_pro3 / A -> D (정답 D) :: .. per bar - J: $2.269... per bar The lowest price per bar is option D with $2.25 per bar. So the answer is D.`
- `math:7709 / upstage_solar_pro3 / D -> F (정답 F) ::  exactly one capital letter naming correct option, no explanation, no punctuation, no other text. So just "F".`

## 예시 — 읽던 답을 버림

- `math:8977 / upstage_solar_pro3 / A -> None (정답 J) :: (fixed per stratum), which is not proportional; could be representative if you think all states are equal? But`
- `math:8977 / upstage_solar_pro3 / A -> None (정답 J) :: , then choose 3% of the customers from each state. - Similar to H but proportionate to total population (if 3%`
- `math:8478 / upstage_solar_pro3 / G -> None (정답 G) ::  I said this equals (2, -3), so 2 - a = 2 => a = 0, and -b = -3 => b = 3, which matches what I found. Now, T(2`
- `math:7695 / upstage_solar_pro3 / J -> None (정답 I) :: f D_3 is order 6, and D_4 is order 8, then the direct product is order 48, as I said. No match. So Statement 2`

## 예시 — 못 읽던 답을 읽음

- `physics:9570 / upstage_solar_pro3 / None -> F (정답 E) :: [F]`
- `psychology:2271 / upstage_solar_pro3 / None -> C (정답 C) :: [C]`
- `math:8859 / anthropic_haiku / None -> B (정답 B) :: π²/49 ≈ 0.2012: $$w = \int_0^{28} \sqrt{1 + 0.2012\cos^2\left(\frac{\pi x}{7}\right)} dx \approx 29.36$$ **B**`
- `math:8766 / anthropic_haiku / None -> B (정답 B) :: $, which separates solutions that become negative ($a < 1.5$) from those always positive ($a \geq 1.5$). **B**`
