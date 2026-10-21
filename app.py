
import io
import random
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types
from streamlit_drawable_canvas import st_canvas
from streamlit_autorefresh import st_autorefresh


# ==========================================
# 기본 설정
# ==========================================

st.set_page_config(
    page_title="AI 캐치마인드",
    page_icon="🎨",
    layout="wide",
)

CSV_PATH = Path(__file__).parent / "keyword.csv"

CATEGORIES = ["동물", "과일", "채소", "사물", "교통수단"]

QUESTION_COUNT = 5
TIME_LIMIT = 60

CANVAS_WIDTH = 700
CANVAS_HEIGHT = 420


# ==========================================
# CSV 단어 데이터 불러오기
# ==========================================

@st.cache_data
def load_keywords():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            "keyword.csv 파일이 app.py와 같은 폴더에 없습니다."
        )

    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

    required_columns = {"category", "keyword"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "CSV 파일에는 category, keyword 열이 필요합니다."
        )

    df = df.dropna(subset=["category", "keyword"]).copy()

    df["category"] = df["category"].astype(str).str.strip()
    df["keyword"] = df["keyword"].astype(str).str.strip()

    df = df[
        (df["category"] != "")
        & (df["keyword"] != "")
    ]

    return df


# ==========================================
# 게임 초기화
# ==========================================

def init_game(category):
    try:
        df = load_keywords()
    except Exception as exc:
        st.error(f"단어 파일을 읽지 못했습니다: {exc}")
        return

    words = (
        df.loc[df["category"] == category, "keyword"]
        .drop_duplicates()
        .tolist()
    )

    if len(words) < QUESTION_COUNT:
        st.error(
            f"{category} 카테고리에 서로 다른 단어가 "
            f"{QUESTION_COUNT}개 이상 필요합니다."
        )
        return

    now = time.time()

    st.session_state.game = {
        "category": category,
        "questions": random.sample(words, QUESTION_COUNT),
        "index": 0,
        "results": [],
        "deadline": now + TIME_LIMIT,
        "finished_question": False,
        "drawing": None,
        "ai_answer": "",
        "ai_error": "",
    }

    st.session_state.screen = "game"


# ==========================================
# 그림 이미지 저장
# ==========================================

def image_to_png_bytes(image):
    buffer = io.BytesIO()

    image.convert("RGB").save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


# ==========================================
# Gemini AI 그림 추측
# ==========================================

def guess_with_gemini(png_bytes):
    api_key = st.secrets.get("GEMINI_API_KEY", "")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY가 없습니다. "
            "Streamlit Secrets를 확인해 주세요."
        )

    client = genai.Client(api_key=api_key)

    image = Image.open(
        io.BytesIO(png_bytes)
    ).convert("RGB")

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=[
            (
                "당신은 캐치마인드 게임의 그림 추측 AI입니다. "
                "제공된 그림이 무엇인지 추측하세요. "
                "한국어 단어 하나만 출력하세요. "
                "설명, 문장, 따옴표, 문장부호는 출력하지 마세요."
            ),
            image,
        ],
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=20,
        ),
    )

    # response와 answer는 반드시 같은 함수 안에 둡니다.
    answer = (response.text or "").strip()

    if answer:
        answer = answer.splitlines()[0].strip()
    else:
        answer = "모르겠음"

    # 답변 앞뒤의 불필요한 따옴표 제거
    answer = answer.strip("\"'「」『』 ")

    return answer[:40] if answer else "모르겠음"


# ==========================================
# 제출한 그림 확정 및 AI 판별
# ==========================================

def submit_drawing(canvas_result, timed_out=False):
    game = st.session_state.game

    # 이미 제출한 문제는 다시 저장하지 않습니다.
    if game["finished_question"]:
        return

    image = None

    if (
        canvas_result is not None
        and canvas_result.image_data is not None
    ):
        try:
            image = Image.fromarray(
                canvas_result.image_data.astype("uint8")
            ).convert("RGB")
        except Exception:
            image = None

    # 그림이 없거나 이미지 변환에 실패한 경우 빈 그림 저장
    if image is None:
        image = Image.new(
            "RGB",
            (CANVAS_WIDTH, CANVAS_HEIGHT),
            "white",
        )

    png_bytes = image_to_png_bytes(image)

    # 그림을 먼저 확정하고 잠급니다.
    game["drawing"] = png_bytes
    game["finished_question"] = True
    game["ai_answer"] = ""
    game["ai_error"] = ""

    try:
        with st.spinner("🤖 AI가 그림을 추측하고 있어요..."):
            game["ai_answer"] = guess_with_gemini(png_bytes)

    except Exception as exc:
        game["ai_answer"] = "판별 실패"
        game["ai_error"] = str(exc)

    # 문제별 결과 기록
    game["results"].append({
        "keyword": game["questions"][game["index"]],
        "drawing": png_bytes,
        "ai_answer": game["ai_answer"],
        "ai_error": game["ai_error"],
        "timed_out": timed_out,
    })


# ==========================================
# 다음 문제로 이동
# ==========================================

def advance_question():
    game = st.session_state.game

    game["index"] += 1

    if game["index"] >= QUESTION_COUNT:
        st.session_state.screen = "results"
        return

    game["deadline"] = time.time() + TIME_LIMIT
    game["finished_question"] = False
    game["drawing"] = None
    game["ai_answer"] = ""
    game["ai_error"] = ""


# ==========================================
# 화면 라우팅 초기화
# ==========================================

if "screen" not in st.session_state:
    st.session_state.screen = "start"


st.title("🎨 AI 캐치마인드")
st.caption(
    "그림을 그리고 제출하면 AI가 정답을 추측합니다. "
    "총 5문제, 문제당 60초!"
)


# ==========================================
# 1. 시작 화면
# ==========================================

if st.session_state.screen == "start":

    st.subheader("어떤 카테고리로 플레이할까요?")

    st.write(
        "카테고리를 선택하면 해당 분야의 단어가 무작위로 출제됩니다. "
        "정답은 모든 문제가 끝난 후 확인할 수 있습니다."
    )

    icons = {
        "동물": "🐾",
        "과일": "🍎",
        "채소": "🥕",
        "사물": "🧸",
        "교통수단": "🚗",
    }

    columns = st.columns(5)

    for column, category in zip(columns, CATEGORIES):
        with column:
            st.markdown(f"## {icons[category]}")
            st.markdown(f"### {category}")

            if st.button(
                f"{category} 시작",
                key=f"start_{category}",
                use_container_width=True,
                type="primary",
            ):
                init_game(category)

                if st.session_state.screen == "game":
                    st.rerun()


# ==========================================
# 2. 게임 화면
# ==========================================

elif st.session_state.screen == "game":

    game = st.session_state.game
    index = game["index"]

    remaining = max(
        0,
        int(game["deadline"] - time.time()),
    )

    st.progress(
        (index + 1) / QUESTION_COUNT,
        text=(
            f"문제 {index + 1} / {QUESTION_COUNT} "
            f"· 카테고리: {game['category']}"
        ),
    )

    st.subheader(f"✏️ 문제 {index + 1}")

    st.info(
        "정답 단어를 그림으로 표현해 보세요! "
        "그림 제출 전에는 정답이 공개되지 않습니다."
    )

    if not game["finished_question"]:

        # 1초마다 화면을 갱신해 타이머를 업데이트합니다.
        st_autorefresh(
            interval=1000,
            key=f"timer_{index}_{int(game['deadline'])}",
        )

        timer_col, tip_col = st.columns([1, 3])

        with timer_col:
            st.metric(
                label="남은 시간",
                value=f"{remaining}초",
            )

        with tip_col:
            st.caption(
                "태블릿에서는 손가락이나 스타일러스를 사용할 수 있습니다."
            )

        stroke_width = st.slider(
            "선 굵기",
            min_value=2,
            max_value=18,
            value=5,
            key=f"width_{index}",
        )

        stroke_color = st.color_picker(
            "선 색상",
            value="#222222",
            key=f"color_{index}",
        )

        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color="#FFFFFF",
            height=CANVAS_HEIGHT,
            width=CANVAS_WIDTH,
            drawing_mode="freedraw",
            display_toolbar=True,
            update_streamlit=True,
            key=f"canvas_{index}",
        )

        st.caption(
            "그림판 도구 모음으로 지우거나 초기화할 수 있습니다. "
            "제출 후에는 그림을 수정할 수 없습니다."
        )

        submit_button = st.button(
            "✅ 그림 제출하기",
            type="primary",
            use_container_width=True,
        )

        # 버튼을 누르거나 제한 시간이 끝나면 제출 처리
        if submit_button or remaining <= 0:

            submit_drawing(
                canvas_result,
                timed_out=(remaining <= 0 and not submit_button),
            )

            st.rerun()

    
else:
    st.success("제출 완료! 그림이 저장되어 잠겼습니다.")

    st.image(
        Image.open(io.BytesIO(game["drawing"])),
        caption="제출한 그림",
        use_container_width=True,
    )

    # AI 답변을 비어 있지 않게 표시
    ai_answer = game.get("ai_answer", "")

    if ai_answer:
        st.subheader(f"🤖 AI의 답: {ai_answer}")
    else:
        st.error("AI 답변이 비어 있습니다.")
        st.write("오류 내용:", game.get("ai_error", "오류 정보 없음"))

    if game.get("ai_error"):
        st.code(game["ai_error"])

    button_label = (
        "다음 문제 ➡️"
        if game["index"] < QUESTION_COUNT - 1
        else "🏁 결과 확인하기"
    )

    if st.button(
        button_label,
        type="primary",
        use_container_width=True,
    ):
        advance_question()
        st.rerun()


        if game["ai_error"]:
            st.warning(
                f"AI 판별 중 오류가 발생했습니다: {game['ai_error']}"
            )

        if index < QUESTION_COUNT - 1:
            button_label = "다음 문제 ➡️"
        else:
            button_label = "🏁 결과 확인하기"

        if st.button(
            button_label,
            type="primary",
            use_container_width=True,
        ):
            advance_question()
            st.rerun()


# ==========================================
# 3. 결과 화면
# ==========================================

elif st.session_state.screen == "results":

    game = st.session_state.game

    st.subheader("🏁 게임 결과")

    st.write(
        f"카테고리: **{game['category']}** · "
        f"완료한 문제: **{len(game['results'])}개**"
    )

    for number, result in enumerate(
        game["results"],
        start=1,
    ):

        with st.container(border=True):

            st.markdown(f"### 문제 {number}")

            image_col, answer_col = st.columns([1, 1])

            with image_col:
                st.image(
                    Image.open(
                        io.BytesIO(result["drawing"])
                    ),
                    caption="내가 그린 그림",
                    use_container_width=True,
                )

            with answer_col:
                st.markdown(
                    f"**🤖 AI의 답:** {result['ai_answer']}"
                )

                st.markdown(
                    f"**🎯 실제 정답:** {result['keyword']}"
                )

                if result["timed_out"]:
                    st.caption("⏰ 시간 종료로 자동 제출되었습니다.")

                if result["ai_error"]:
                    st.caption(
                        f"AI 오류: {result['ai_error']}"
                    )

    if st.button(
        "🔄 처음으로 돌아가기",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.pop("game", None)
        st.session_state.screen = "start"
        st.rerun()
