#!/usr/bin/env bash
# 측정 서버 초기 설정. Ubuntu 24.04 LTS 기준이고 서버에서 직접 돌린다.
#
#   ssh -i <키> ubuntu@<고정IP>
#   bash ~/llm_trafic_router/V2/deploy/bootstrap.sh
#
# 러너 코드는 이 스크립트가 돌기 전에 로컬에서 rsync로 올려 둔다.
# 절차는 같은 디렉터리의 README.md에 있다.
set -euo pipefail

REPO="$HOME/llm_trafic_router"
RUNNER="$REPO/V2/runner"

echo "== 1. 패키지 =="
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip chrony rsync

echo
echo "== 2. 스왑 1GB =="
# 램이 1GB뿐이라 pip 설치나 재시도가 몰릴 때 OOM으로 프로세스가 죽을 수 있다.
# 21일 무인 실행에서 그것은 그날의 완주 실패를 뜻한다.
if [ ! -f /swapfile ]; then
  sudo fallocate -l 1G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile >/dev/null
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
  echo "스왑 생성함"
else
  echo "이미 있음"
fi

echo
echo "== 3. 시각 =="
# 슬롯 라벨이 UTC 시각으로 정해지므로 시계가 틀리면 조건 배정이 틀린다.
# 로컬 시간대를 UTC로 고정해 로그와 슬롯 인덱스의 기준을 하나로 맞춘다.
sudo timedatectl set-timezone UTC
sudo systemctl enable --now chrony
sleep 2
timedatectl show -p Timezone -p NTPSynchronized
chronyc tracking | grep -E 'Reference ID|System time|Stratum' || true

echo
echo "== 4. 파이썬 환경 =="
python3 -m venv "$REPO/.venv"
"$REPO/.venv/bin/pip" install -q --upgrade pip
"$REPO/.venv/bin/pip" install -q -r "$RUNNER/requirements-run.txt"
"$REPO/.venv/bin/python" -c "import requests, dotenv; print('의존성 설치됨', requests.__version__)"

echo
echo "== 5. 산출물 디렉터리 =="
# rsync에서 outputs/를 제외한다. 보정 로그 46MB가 그 안에 있고 서버는 그것을
# 읽지 않기 때문이다. 다만 러너와 스모크 테스트는 이 경로에 결과를 쓰므로
# 빈 디렉터리는 있어야 한다.
mkdir -p "$RUNNER/outputs"
echo "$RUNNER/outputs"

echo
echo "== 6. 배치 확인 =="
for f in "$REPO/.env" "$RUNNER/data/item_bank.json" "$RUNNER/capabilities.json"; do
  if [ -f "$f" ]; then echo "  있음  $f"; else echo "  없음  $f"; fi
done

echo
echo "== 7. 키 점검 =="
cd "$RUNNER" && "$REPO/.venv/bin/python" check_env.py

echo
echo "다음: 데이터센터 IP 스모크 테스트"
echo "  cd $RUNNER && $REPO/.venv/bin/python smoke_test.py"
