# 1. LaunchAgents 폴더에 복사
cp /Users/nojaegyeong/Documents/GitHub/llm_trafic_router/com.llmbench.experiment.plist ~/Library/LaunchAgents/

# 2. launchd에 등록 및 시작
launchctl load ~/Library/LaunchAgents/com.llmbench.experiment.plist

# 3. 상태 확인
launchctl list | grep llmbench

# 4. 종료
launchctl load ~/Library/LaunchAgents/com.llmbench.experiment.plist