# 보정 패스 세 배치를 순서대로 돌린다 (설계서 6.1절).
#
# 각 배치는 --wait으로 자기 저부하 시간대가 열릴 때까지 기다린다.
# 야간 배치(cn.kr, KST 23~07시)를 앞에 두는 것이 핵심이다. us를 먼저
# 두면 다음 날 10시까지 기다리느라 그날 밤을 통째로 버린다.
#
# 실행:
#     powershell -ExecutionPolicy Bypass -File run_calibration.ps1
#
# 이 파일은 반드시 UTF-8 BOM으로 저장한다. PowerShell 5.1은 BOM이 없으면
# .ps1을 cp949로 읽고, 한글 3바이트가 쪼개지며 따옴표를 잡아먹어 구문
# 분석이 통째로 깨진다.
#
# 중단해도 안전하다. 러너는 성공한 콜만 재개 키로 세므로, 같은 명령을
# 다시 치면 남은 콜부터 이어간다.

Set-Location $PSScriptRoot

# 콘솔 입출력도 UTF-8로 맞춘다. config.py가 파이썬 쪽 stdout을 손보지만
# PowerShell의 Tee-Object는 별개다.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$batches = @(
    [pscustomobject]@{ Region = 'cn'; Label = '중국 (DeepSeek, Qwen)'; Extra = @() }
    [pscustomobject]@{ Region = 'kr'; Label = '한국 (Upstage, HyperCLOVA)'; Extra = @() }
    [pscustomobject]@{ Region = 'us'; Label = '미국 (OpenAI, Google, Anthropic + 앵커)'; Extra = @('--min-remaining', '2') }
)

New-Item -ItemType Directory -Force -Path 'outputs' | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmm'
$logFile = Join-Path 'outputs' "calibration_run_$stamp.log"
$results = @()

function Write-Both([string]$Text) {
    Write-Host $Text
    Add-Content -Path $logFile -Value $Text -Encoding UTF8
}

Write-Both ''
Write-Both ('=' * 68)
Write-Both ('보정 패스 시작  ' + (Get-Date -Format 'yyyy-MM-dd HH:mm'))
Write-Both ('로그: ' + $logFile)
Write-Both ('=' * 68)

foreach ($b in $batches) {
    Write-Both ''
    Write-Both ('-' * 68)
    Write-Both ('[' + $b.Region + '] ' + $b.Label + '   ' + (Get-Date -Format 'HH:mm'))
    Write-Both ('-' * 68)

    $argv = @('run', 'calibrate.py', '--region', $b.Region, '--k', '5', '--wait') + $b.Extra
    $started = Get-Date

    & uv @argv 2>&1 | Tee-Object -FilePath $logFile -Append

    $code = $LASTEXITCODE
    $mins = [int]((Get-Date) - $started).TotalMinutes

    if ($code -eq 0) {
        $results += [pscustomobject]@{ Region = $b.Region; Status = '완료'; Minutes = $mins }
        Write-Both ('[' + $b.Region + '] 완료 - ' + $mins + '분')
    }
    else {
        $results += [pscustomobject]@{ Region = $b.Region; Status = ('실패 exit ' + $code); Minutes = $mins }
        Write-Both ('[' + $b.Region + '] 실패 - exit ' + $code + ', ' + $mins + '분')
        # 다음 배치를 계속 돌린다. 지역별로 프로바이더가 다르므로 한쪽
        # 실패가 다른 쪽 실패를 뜻하지 않고, 여기서 멈추면 오늘 밤
        # 남은 시간대를 통째로 버리게 된다.
        Write-Both '     (다음 배치를 계속 진행한다. 실패한 지역은 나중에 다시 돌리면 이어간다.)'
    }
}

Write-Both ''
Write-Both ('=' * 68)
Write-Both ('보정 패스 종료  ' + (Get-Date -Format 'yyyy-MM-dd HH:mm'))
foreach ($r in $results) {
    Write-Both ('  ' + $r.Region.PadRight(4) + $r.Status.PadRight(16) + $r.Minutes + '분')
}

$failed = @($results | Where-Object { $_.Status -ne '완료' })
if ($failed.Count -gt 0) {
    Write-Both ''
    Write-Both '실패한 배치가 있다. 해당 지역만 다시 돌리면 남은 콜부터 이어간다:'
    foreach ($f in $failed) {
        Write-Both ('  uv run calibrate.py --region ' + $f.Region + ' --k 5 --wait')
    }
    Write-Both ''
    Write-Both '세 배치가 모두 끝나야 다음 단계로 갈 수 있다.'
}
else {
    Write-Both ''
    Write-Both '세 배치 모두 완료. 다음 단계:'
    Write-Both '  uv run select_bank.py    # 문항 은행 300 + 노이즈 바닥선'
    Write-Both '  uv run budget.py         # 실측 기반 투영, 지출 상한 확정'
}
Write-Both ('=' * 68)

if ($failed.Count -gt 0) { exit 1 }
