import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="LLM Routing Monitor", layout="wide")
st.title("📊 LLM Dynamic Routing Real-time Monitor")

# 1. 데이터 로드
@st.cache_data(ttl=600) # 10분마다 데이터 새로고침
def load_data():
    if os.path.exists("benchmark_results.csv"):
        df = pd.read_csv("benchmark_results.csv")
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d_%H%M%S')
        return df
    return None

df = load_data()

if df is not None:
    # 2. 상단 요약 지표 (Metrics)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", len(df))
    col2.metric("Avg TPS (GPT-4o)", round(df[df['provider']=='openai']['tps'].mean(), 2))
    col3.metric("Overall Accuracy", f"{round(df['is_correct'].mean()*100, 1)}%")

    # 3. 시간대별 TPS 추이 (속도 변화 감지)
    st.subheader("Time-series TPS (Speed Analysis)")
    fig_tps = px.line(df, x='timestamp', y='tps', color='provider', 
                      title="Is the model getting faster? (Potential Routing Signal)")
    st.plotly_chart(fig_tps, use_container_width=True)

    # 4. 속도 vs 정확도 산점도 (핵심 증거)
    st.subheader("TPS vs Accuracy Correlation")
    st.write("라우팅이 발생하면 우측 하단(속도는 빠르나 정확도는 낮음)에 점들이 찍힙니다.")
    fig_corr = px.scatter(df, x='tps', y='is_correct', color='provider', 
                         facet_col='difficulty', hover_data=['timestamp'])
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.warning("아직 수집된 데이터가 없습니다. benchmark_results.csv 파일을 확인해주세요.")