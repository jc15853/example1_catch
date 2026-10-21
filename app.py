
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

st.set_page_config(
    page_title="AI 캐치마인드",
    page_icon="🎨",
    layout="wide",
)

CSV_PATH = Path(__file__).with_name("keyword.csv")
CATEGORIES = ["동물", "과일", "채소", "사물", "교통수단"]
QUESTION_COUNT = 5
TIME_LIMIT = 60
CANVAS_WIDTH = 700
CANVAS_HEIGHT = 420


def load_keywords():
    if not CSV_PATH.exists():
        st.error("keyword.csv 파일이 app.py와 같은 폴더에 있어야 합니다.")
        st.stop()

    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

    if not {"category", "keyword"}.issubset(df.columns):
        st.error("CSV에는 category, keyword 열이 필요합니다.")
        st.stop()

    df = df.dropna(subset=["category", "keyword"]).copy()
    df["category"] = df["category"].astype(str).str.strip()
    df["keyword"] = df["keyword"].astype(str).str.strip()

    return df[
        (df["category"] != "") & (df["keyword"] != "")
    ]


def init_game(category):
    df = load_keywords()
    words = (
        df.loc[df["category"] == category, "keyword"]
        .drop_duplicates()
        .tolist()
    )

    if len(words) < QUESTION_COUNT:
        st.error("해당 카테고리에 단어가 5개 이상 필요합니다.")
        st.stop()

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


def image_to_png_bytes(image):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def guess_with_gemini(png_bytes):
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except Exception as exc:
        raise RuntimeError(
            "Streamlit Secrets에 GEMINI_API_KEY를 설정해 주세요."
        ) from exc

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY가 비어 있습니다.")

    client = genai.Client(api_key=api_key)
    image = Image.open(io.BytesIO(png_bytes)).convert("RGB")

    
response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=[
        (
            "그림이 무엇인지 추측하세요. "
            "한국어 한 단어만 출력하세요. "
            "설명이나 문장부호는 출력하지 마세요."
        ),
        image,
    ],
    config=types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=20,
    ),
)


    answer = (response.text or "").strip()
    answer = answer.splitlines()[0].strip() if answer else "모르겠음"
    return answer[:40]


def submit_drawing(canvas_result, timed_out=False):
    game = st.session_state.game

    if game["finished_question"]:
        return

    image = None

    if canvas_result is not None and canvas_result.image_data is not None:
        try:
            image = Image.fromarray(
                canvas_result.image_data.astype("uint8")
            ).convert("RGB")
        except Exception:
            image = None

    if image is None:
        image = Image.new(
            "RGB",
            (CANVAS_WIDTH, CANVAS_HEIGHT),
            "white",
        )

    png_bytes = image_to_png_bytes(image)

    # 제출 시 이미지 데이터를 고정하여 저장
    game["drawing"] = png_bytes
    game["finished_question"] = True

    try:
        with st.spinner("AI가 그림을 추측하고 있어요..."):
            game["ai_answer"] = guess_with_gemini(png_bytes)
    except Exception as exc:
        game["ai_answer"] = "판별 실패"
        game["ai_error"] = str(exc)

    game["results"].append({
        "keyword": game["questions"][game["index"]],
        "drawing": png_bytes,
        "ai_answer": game["ai_answer"],
        "ai_error": game["ai_error"],
        "timed_out": timed_out,
    })


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


# 타이머를 갱신하기 위한 자동 새로고침
from streamlit_autorefresh import st_autorefresh

st.title("🎨 AI 캐치마인드")
st.caption("60초 안에 그림을 그리고 AI의 추측을 확인하세요!")

if "screen" not in st.session_state:
    st.session_state.screen = "start"


# ---------------- 시작 화면 ----------------
if st.session_state.screen == "start":
    st.subheader("카테고리를 선택하세요")
    st.write("선택한 카테고리에서 무작위로 5문제가 출제됩니다.")

    icons = {
        "동물": "🐾",
        "과일": "🍎",
        "채소": "🥕",
        "사물": "🧸",
        "교통수단": "🚗",
    }

    cols = st.columns(5)

    for col, category in zip(cols, CATEGORIES):
        with col:
            st.markdown(f"## {icons[category]}")
            st.write(category)

            if st.button(
                f"{category} 시작",
                key=f"start_{category}",
                use_container_width=True,
            ):
                init_game(category)
                st.rerun()


# ---------------- 게임 화면 ----------------
elif st.session_state.screen == "game":
    game = st.session_state.game
    index = game["index"]

    remaining = max(0, int(game["deadline"] - time.time()))

    st.progress(
        (index + 1) / QUESTION_COUNT,
        text=f"문제 {index + 1} / {QUESTION_COUNT} · {game['category']}",
    )
    st.subheader(f"문제 {index + 1}")
    st.info("정답을 그림으로 표현해 보세요!")

    if not game["finished_question"]:
        st_autorefresh(
            interval=1000,
            key=f"timer_{index}_{int(game['deadline'])}",
        )

        st.metric("남은 시간", f"{remaining}초")
        st.caption(
            "태블릿에서는 손가락이나 스타일러스로 그릴 수 있습니다."
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
            "#222222",
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

        submitted = st.button(
            "그림 제출하기",
            type="primary",
            use_container_width=True,
        )

        if submitted or remaining <= 0:
            submit_drawing(
                canvas_result,
                timed_out=(remaining <= 0 and not submitted),
            )
            st.rerun()

    else:
        st.success("제출 완료! 그림을 수정할 수 없습니다.")

        st.image(
            Image.open(io.BytesIO(game["drawing"])),
            caption="제출한 그림",
            use_container_width=True,
        )
        st.write(f"**AI의 답:** {game['ai_answer']}")

        if game["ai_error"]:
            st.warning(game["ai_error"])

        button_label = (
            "다음 문제"
            if index < QUESTION_COUNT - 1
            else "결과 보기"
        )

        if st.button(
            button_label,
            type="primary",
            use_container_width=True,
        ):
            advance_question()
            st.rerun()


# ---------------- 결과 화면 ----------------
elif st.session_state.screen == "results":
    game = st.session_state.game

    st.subheader("🏁 게임 결과")
    st.write(
        f"카테고리: **{game['category']}** · "
        f"총 {len(game['results'])}문제"
    )

    for i, result in enumerate(game["results"], start=1):
        with st.container(border=True):
            st.markdown(f"### 문제 {i}")

            image_col, answer_col = st.columns([1, 1])

            with image_col:
                st.image(
                    Image.open(io.BytesIO(result["drawing"])),
                    caption="내가 그린 그림",
                    use_container_width=True,
                )

            with answer_col:
                st.markdown(f"**AI의 답:** {result['ai_answer']}")
                st.markdown(f"**실제 정답:** {result['keyword']}")

                if result["timed_out"]:
                    st.caption("시간 종료로 자동 제출됨")

                if result["ai_error"]:
                    st.caption(f"AI 오류: {result['ai_error']}")

    if st.button(
        "처음으로 돌아가기",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.screen = "start"
        st.session_state.pop("game", None)
        st.rerun()
