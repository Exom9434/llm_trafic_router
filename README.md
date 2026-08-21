# LLM API Latency: Peak vs. Off-Peak Hours

**Observing LLM API Latency During Peak and Off-Peak Hours: A 28-Day Log**

A pilot study benchmarking four commercial LLM APIs (OpenAI, Anthropic, Google, Alibaba Qwen) across peak and off-peak US business hours over 28 days, measuring TTFT, TPS, and MMLU-Pro accuracy.

📄 [Read the report (PDF)](report/main.pdf)

## Providers

| Provider | Model |
|----------|-------|
| OpenAI | gpt-4o-mini |
| Anthropic | claude-haiku-4-5 |
| Google | gemini-flash-lite |
| Alibaba Qwen | qwen3.5-flash |

## Key Findings

- US-hosted providers (OpenAI, Anthropic, Google) show **+4–37% higher TTFT** during peak hours
- Qwen shows the opposite pattern — faster during US peak, consistent with cross-regional load balancing
- Answer accuracy shows **no significant change** across peak/off-peak for any provider

## Repository Structure

```
experiment_api.py   # Benchmarking script
report/main.tex     # LaTeX source
report/main.pdf     # Compiled report
analysis/           # Analysis scripts
data/               # Collected CSVs
```
