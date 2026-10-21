import time
import io
import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from google import genai
from google.genai import types

# Page Config
st.set_page_config(
    page_title="추상화 드로잉 퀴즈 - 정보 교과",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI styling
st.markdown("""
<style>
    /* Theme Setup */
    .main {
        background-color: #f8fafc;
    }
    .stAppHeader {
        background-color: rgba(255, 255, 255, 0.8);
    }
    .title-box {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        padding: 24px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .concept-card {
        background-color: #eff6ff;
        border-left: 5px solid #3b82f6;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 20px;
        color: #1e3a8a;
    }
    .timer-badge {
        font-size: 24px;
        font-weight: bold;
        color: #ef4444;
        background-color: #fee2e2;
        padding: 8px 16px;
        border-radius: 20px;
        display: inline-block;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        margin-bottom: 16px;
    }
    .success-tag {
        color: #15803d;
        background-color: #dcfce7;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
    }
    .fail-tag {
        color: #b91c1c;
        background-color: #fee2e2;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# API Key Validation
api_key = st.sidebar.text_input("Gemini API Key 입력", type="password")
if not api_key:
    st.sidebar.warning("⚠️ 사이드바에 Gemini API 키를 입력하셔야 AI 채점 기능을 이용하실 수 있습니다.")

# Load Keywords
@st.cache_data
def load_keywords():
    try:
        return pd.read_csv("keyword.csv")
    except Exception as e:
        st.error(f"keyword.csv 파일을 읽을 수 없습니다: {e}")
        return pd.DataFrame(columns=["category", "keyword"])

df_keywords = load_keywords()

# Initialize Session State
if "game_state" not in st.session_state:
    st.session_state.game_state = "START"  # START, PLAYING, EVALUATING, FINAL
if "quiz_pool" not in st.session_state:
    st.session_state.quiz_pool = []
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "pass_count" not in st.session_state:
    st.session_state.pass_count = 2
if "results" not in st.session_state:
    st.session_state.results = []
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "current_eval" not in st.session_state:
    st.session_state.current_eval = None

# Header Banner
st.markdown("""
<div class="title-box">
    <h1 style='margin:0; font-size: 28px;'>🎨 중1 정보 - '추상화' 드로잉 퀴즈</h1>
    <p style='margin:8px 0 0 0; opacity: 0.9;'>복잡한 불필요한 디테일은 빼고! 핵심 특징만 살려 대상을 표현해 보세요.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. 시작 화면
# -----------------------------------------------------------------------------
if st.session_state.game_state == "START":
    st.markdown("""
    <div class="concept-card">
        <strong>💡 개념 정리: 추상화(Abstraction)란?</strong><br>
        복잡한 문제나 대상에서 <u>불필요한 세부 사항(디테일)은 제거</u>하고, <u>꼭 필요한 핵심 특징만 추출</u>하여 단순하게 표현하는 문제 해결 기법입니다.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("⚙️ 퀴즈 설정")
        num_questions = st.slider("풀고 싶은 문제 수 선택", min_value=1, max_value=20, value=5)
        
        categories = df_keywords["category"].unique().tolist()
        st.write("📂 **카테고리 선택**")
        selected_cat = st.radio("원하는 카테고리를 선택하세요:", categories, horizontal=True)

    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        if st.button("🚀 게임 시작하기", use_container_width=True, type="primary"):
            filtered = df_keywords[df_keywords["category"] == selected_cat]
            if len(filtered) == 0:
                st.error("해당 카테고리에 키워드가 없습니다.")
            else:
                sample_size = min(num_questions, len(filtered))
                selected_pool = filtered.sample(n=sample_size).to_dict("records")
                
                st.session_state.quiz_pool = selected_pool
                st.session_state.current_idx = 0
                st.session_state.pass_count = 2
                st.session_state.results = []
                st.session_state.start_time = time.time()
                st.session_state.game_state = "PLAYING"
                st.rerun()

# -----------------------------------------------------------------------------
# 2. 게임(드로잉) 화면
# -----------------------------------------------------------------------------
elif st.session_state.game_state == "PLAYING":
    current_quiz = st.session_state.quiz_pool[st.session_state.current_idx]
    
    # 타이머 계산 (60초)
    elapsed = time.time() - st.session_state.start_time
    remaining = max(0, 60 - int(elapsed))

    # 상단 정보 표시
    top_col1, top_col2, top_col3 = st.columns([1, 2, 1])
    with top_col1:
        st.markdown(f"### 📌 문제 {st.session_state.current_idx + 1} / {len(st.session_state.quiz_pool)}")
    with top_col2:
        st.markdown(f"<h2 style='text-align: center; color: #1e40af;'>제시어: [{current_quiz['keyword']}]</h2>", unsafe_allow_html=True)
    with top_col3:
        st.markdown(f"<div style='text-align: right;'><span class='timer-badge'>⏱️ {remaining}초</span></div>", unsafe_allow_html=True)

    if remaining <= 0:
        st.warning("⏰ 시간이 초과되었습니다! 자동으로 제출 평가로 넘어갑니다.")

    # 캔버스 툴바
    st.markdown("---")
    tool_col1, tool_col2 = st.columns([1, 2])
    with tool_col1:
        color_option = st.radio("🎨 색상 선택", ["검정", "빨강", "파랑", "초록", "커스텀"], horizontal=True)
        if color_option == "검정":
            stroke_color = "#000000"
        elif color_option == "빨강":
            stroke_color = "#ef4444"
        elif color_option == "파랑":
            stroke_color = "#3b82f6"
        elif color_option == "초록":
            stroke_color = "#22c55e"
        else:
            stroke_color = st.color_picker("커스텀 색상", "#000000")

    with tool_col2:
        stroke_width = st.slider("🖌️ 선 두께 조절", 1, 25, 5)

    # 캔버스 컴포넌트
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color="#ffffff",
        height=380,
        width=700,
        drawing_mode="freedraw",
        key=f"canvas_{st.session_state.current_idx}"
    )

    # 하단 버튼 액션
    btn_col1, btn_col2, _ = st.columns([2, 2, 3])
    
    with btn_col1:
        if st.button("✨ 추상화 결과 제출하기", type="primary", use_container_width=True):
            if canvas_result.image_data is not None:
                img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
                # AI 채점 로직 호출
                st.session_state.current_canvas
