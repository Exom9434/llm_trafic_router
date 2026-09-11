#!/usr/bin/env bash
# 러너 코드를 측정 서버로 올린다. 맥에서 돌린다.
#
#   bash "V2/deploy/push.sh"
#
# 서버가 실행 시점에 읽는 것만 올린다. 저장소에는 46MB짜리 보정 로그와
# 3월 벤치마크 CSV가 들어 있어 통째로 올릴 이유가 없다.
set -euo pipefail

IP="${IP:-3.34.88.19}"
KEY="${KEY:-$HOME/.ssh/lightsail-seoul.pem}"
LOCAL="${LOCAL:-$HOME/Documents/문서 - 노재경의 MacBook Pro/GitHub/llm_trafic_router}"
REMOTE=ubuntu@"$IP":llm_trafic_router

ssh -i "$KEY" ubuntu@"$IP" 'mkdir -p ~/llm_trafic_router/V2/runner/outputs'

rsync -avz --delete -e "ssh -i $KEY" \
  --exclude '__pycache__' --exclude '.venv' --exclude 'outputs' --exclude 'capabilities.json' \
  "$LOCAL/V2/runner/" "$REMOTE/V2/runner/"

rsync -avz -e "ssh -i $KEY" "$LOCAL/V2/deploy/" "$REMOTE/V2/deploy/"
rsync -avz -e "ssh -i $KEY" "$LOCAL/.env"      "$REMOTE/.env"

# 예산 계획만 outputs에서 따로 올린다. 러너가 지출 상한과 하루 예약을
# 여기서 읽는데, 이 파일을 만드는 보정 로그는 서버에 올리지 않는다.
rsync -avz -e "ssh -i $KEY" \
  "$LOCAL/V2/runner/outputs/budget_plan.json" "$REMOTE/V2/runner/outputs/"

echo
echo "올렸다. 다음:"
echo "  ssh -i $KEY ubuntu@$IP"
echo "  bash ~/llm_trafic_router/V2/deploy/bootstrap.sh"
