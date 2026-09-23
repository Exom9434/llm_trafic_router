# 측정 서버 배포

본실험 측정 지점을 윈도우 랩탑에서 한국 리전 클라우드 서버로 옮긴다.
소속 기관 절전 계획으로 야간에 전원과 네트워크가 끊길 수 있고, 그러면 그
시각의 슬롯이 빠져 그날이 완주 실패가 되기 때문이다.

측정 지점을 한국에 두는 이유는 보정 측정이 한국에서 이뤄졌기 때문이다.
본실험이 다른 지역에서 돌면 문항 은행과 기저 자기일관성의 기준선이
측정 지점 변경과 섞인다.

## 인스턴스

AWS Lightsail 서울 리전(ap-northeast-2), $7 등급.
2 vCPU / 1GB RAM / 40GB SSD / 2TB 전송.

Vultr 서울이 $5로 더 싸지만 고르지 않았다. 서버 비용은 API 예산 205달러
옆에서 차이가 없는 반면, 21일 무인 실행에서 가장 확률이 높은 실패는
하드웨어가 아니라 계정이나 결제가 막혀 인스턴스가 멎는 쪽이다.

사양은 이 정도면 남는다. 하는 일이 호출을 보내고 JSONL을 쌓는 것뿐이고
본실험 로그가 300MB 안팎이다.

## 콘솔에서 만들 때

- 리전은 서울(ap-northeast-2), 가용 영역은 A
- 플랫폼은 Linux/Unix, 블루프린트는 OS Only의 Ubuntu 24.04 LTS
- 요금제는 $7 등급
- SSH 키는 새로 만들어 pem 파일을 내려받는다
- 만든 뒤 고정 IP를 붙인다. 붙여 두면 요금이 없고, 재부팅에 IP가 바뀌면
  접속 경로를 매번 다시 찾아야 한다

## 코드 올리기

저장소에는 46MB짜리 보정 로그와 3월 벤치마크 CSV가 들어 있어 통째로 올릴
이유가 없다. 서버가 실행 시점에 읽는 것은 넷뿐이다.

- `.env` (키 8개)
- `V2/runner/` 코드
- `V2/runner/data/item_bank.json` (문항 은행)
- `V2/runner/capabilities.json` (스모크 테스트 결과, 서버에서 다시 생성한다)

로컬에서:

```bash
LOCAL="/Users/nojaegyeong/Documents/문서 - 노재경의 MacBook Pro/GitHub/llm_trafic_router"
IP=<고정 IP>
KEY=~/.ssh/lightsail-seoul.pem
chmod 400 $KEY

ssh -i $KEY ubuntu@$IP 'mkdir -p ~/llm_trafic_router/V2'

rsync -avz -e "ssh -i $KEY" \
  --exclude '__pycache__' --exclude '.venv' --exclude 'outputs' \
  "$LOCAL/V2/runner/" ubuntu@$IP:~/llm_trafic_router/V2/runner/

rsync -avz -e "ssh -i $KEY" \
  "$LOCAL/V2/deploy/" ubuntu@$IP:~/llm_trafic_router/V2/deploy/

rsync -avz -e "ssh -i $KEY" \
  "$LOCAL/.env" ubuntu@$IP:~/llm_trafic_router/.env
```

경로에 공백과 한글이 들어 있으므로 큰따옴표를 벗기지 않는다.

`outputs/`를 뺀 것은 보정 로그 46MB가 그 안에 있고 서버가 그것을 읽지 않기
때문이다. 예산 가드가 쓰는 `budget_plan.json`은 서버에서 다시 만든다.

## 서버에서

```bash
ssh -i $KEY ubuntu@$IP
bash ~/llm_trafic_router/V2/deploy/bootstrap.sh
```

패키지 설치, 스왑 1GB, 시간대 UTC 고정과 시각 동기, 가상환경, 키 점검까지
한 번에 돈다.

## 그다음에 바로 할 것

데이터센터 IP에서 제공사가 정상 응답하는지 확인한다. 가정용 회선과 rate
limit 정책이 다를 수 있고, 여기서 걸리면 제공사 구성을 바꿔야 하므로 다른
설정보다 앞에 둔다.

```bash
cd ~/llm_trafic_router/V2/runner
~/llm_trafic_router/.venv/bin/python smoke_test.py
```

결과가 로컬의 `outputs/smoke_test.md`와 어긋나는 항목이 있으면 그것이
측정 지점 이동의 효과다. 기록해 두고 사전등록 본문에 반영한다.

## 본실험 띄우기

스모크 테스트가 통과하면 순서는 이렇다.

먼저 예산 계획을 서버에서 다시 만든다. 러너가 `outputs/budget_plan.json`을
읽어 지출 상한과 하루 예약을 잡기 때문이다. 이 파일은 보정 로그에서 실측
토큰을 뽑아 만드는데 그 로그를 서버에 올리지 않았으므로, 로컬에서 만든
것을 올리는 편이 빠르다.

```bash
rsync -avz -e "ssh -i $KEY" \
  "$LOCAL/V2/runner/outputs/budget_plan.json" \
  ubuntu@$IP:~/llm_trafic_router/V2/runner/outputs/
```

다음으로 배선을 확인한다. API 키 없이 파이프라인 전체를 도는 점검과,
일정을 눈으로 보는 예행이다.

```bash
cd ~/llm_trafic_router/V2/runner
V=~/llm_trafic_router/.venv/bin/python
$V check_env.py
$V selftest.py
$V experiment.py --dry-run
```

`--dry-run`에서 확인할 것은 셋이다. 첫 사흘 표에서 묶음 G0·G1·G2가 여덟
시각을 한 번씩 통과하는지, 조건 라벨 표에서 미국만 5/7이고 중국·한국이
7/7인지, 콜 수와 투영 비용이 예산 상한 안인지다.

그다음 슬롯 하나만 실제로 돌려 본다. 21일을 걸기 전에 진짜 API로 한 바퀴
도는 것이 이 단계의 목적이다.

```bash
$V experiment.py --max-slots 1 --log outputs/rehearsal_calls.jsonl
```

**로그를 따로 받는 것이 중요하다.** 사전등록을 아직 내지 않았다면 이
리허설은 등록 이전의 측정이다. 본실험 로그에 섞어 두면 "등록 전에 결과를
엿보지 않았다"는 방어가 그 슬롯만큼 흐려진다. 보정 패스가 저부하에서만
돌았다는 사실을 등록문에 적어 그 의심을 차단하기로 한 것과 같은 이유다
(설계서 10절).

등록을 마친 뒤에 시작하는 실행이 `outputs/main_calls.jsonl`의 첫 줄이
된다. 리허설 파일은 배선 확인의 기록으로 남겨 두고 분석에 넣지 않는다.

리허설을 띄우는 시각도 골라야 한다. `SLOT_CATCHUP_MINUTES`가 30이라 슬롯
시작 30분 안에 뜨면 러너가 그 슬롯을 놓친 것으로 보고 바로 따라잡는다.
슬롯을 발사하지 않고 배선만 보려면 슬롯 시작 30분 뒤부터 다음 슬롯 사이에
띄운다. 슬롯은 09·12·15·18·21·24·03·06시(한국시간)다.

**`--log`는 콜 로그만 바꾼다.** 상태 파일 셋은 경로가 고정이라 리허설이
본실험 자리에 쓴다.

```
outputs/experiment_state.json   슬롯 번호의 기준점 (start_day, run_id)
outputs/slot_status.jsonl       발사한 (모델, 슬롯) 쌍
outputs/day_status.jsonl        날짜별 완주 판정
```

`day_status.jsonl`이 특히 문제다. 여기 적힌 완주일을 `complete_day_counts`가
세고, 그것이 21일을 채웠는지 판단하는 근거다. 리허설 흔적이 남으면 본실험이
하루를 공짜로 얻는다. 본실행 직전에 서버와 로컬 양쪽에서 옮겨 둔다.

```bash
cd ~/llm_trafic_router/V2/runner/outputs
mkdir -p rehearsal_state_backup
mv experiment_state.json slot_status.jsonl day_status.jsonl rehearsal_state_backup/
```

지우지 않고 옮기는 것은 리허설 기록도 배선 확인의 증거여서다. 드롭인으로
`ExecStart`를 덮어 두었다면 그것도 같이 걷는다.

```bash
sudo rm -r /etc/systemd/system/llm-experiment.service.d
sudo systemctl daemon-reload
```

마지막으로 서비스로 올린다.

```bash
sudo cp ~/llm_trafic_router/V2/deploy/llm-experiment.service /etc/systemd/system/
sudo cp ~/llm_trafic_router/V2/deploy/llm-experiment-monitor.* /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llm-experiment
sudo systemctl enable --now llm-experiment-monitor.timer

journalctl -u llm-experiment -f
```

## 실행 중에 보는 것

```bash
journalctl -u llm-experiment -n 100          # 최근 로그
systemctl status llm-experiment              # 살아 있는지
~/llm_trafic_router/.venv/bin/python monitor.py --summary   # 날짜별 요약
wc -l ~/llm_trafic_router/V2/runner/outputs/main_calls.jsonl
cat ~/llm_trafic_router/V2/runner/outputs/day_status.jsonl  # 완주 여부
```

알림을 받으려면 `.env`에 `NOTIFY_WEBHOOK`을 넣는다. Slack·Discord·ntfy처럼
JSON을 받는 주소면 된다. 비어 있으면 감시는 journal에만 남으므로 사람이
들여다봐야 한다.

## 결과 가져오기

로그가 300MB 안팎까지 자란다. 중간 백업은 날마다 받는 것이 낫다. 21일치를
마지막에 한 번 받다가 실패하면 되돌릴 방법이 없다.

```bash
rsync -avz -e "ssh -i $KEY" \
  ubuntu@$IP:~/llm_trafic_router/V2/runner/outputs/ \
  "$LOCAL/V2/runner/outputs_server/"
```
