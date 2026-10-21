
import io
import random
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from streamlit_autorefresh import st_autorefresh
from google import genai


# ----------------------------
# 기본 설정
# ----------------------------
st.set_page_config(
    page_title="AI 캐치마인드",
    page_icon="🎨",
    layout="wide",
)

CSV_PATH = Path(__file__).parent / "keyword.csv"
QUESTION_COUNT = 5
TIME_LIMIT = 60

CATEGORIES = ["동물", "과일", "채소", "사물", "교통수단"]


# ----------------------------
# 단어 불러오기
# ----------------------------
@st.cache_data
def load_keywords():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"keyword.csv 파일을 찾을 수 없습니다: {CSV_PATH}"
        )

    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

    required_columns = {"category", "keyword"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            "keyword.csv에는 category, keyword 열이 필요합니다."
        )

    df = df.dropna(subset=["category", "keyword"]).copy()
    df["category"] = df["category"].astype(str).str.strip()
    df["keyword"] = df["keyword"].astype(str).str.strip()
    df = df[(df["category"] != "") & (df["keyword"] != "")]
    return df


# ----------------------------
# 게임 상태
# ----------------------------
def init_game(category):
    df = load_keywords()
    available = df[df["category"] == category]["keyword"].tolist()

    if len(available) < QUESTION_COUNT:
        st.error(
            f"{category} 카테고리의 단어가 부족합니다. "
            f"최소 {QUESTION_COUNT}개를 넣어 주세요."
        )
        st.stop()

    questions = random.sample(available, QUESTION_COUNT)

    st.session_state.game = {
        "category": category,
        "questions": questions,
        "index": 0,
        "results": [],
        "deadline": time.time() + TIME_LIMIT,
        "finished_question": False,
        "drawing": None,
        "ai_answer": "",
        "ai_error": "",
        "timed_out": False,
    }


def reset_question():
    game = st.session_state.game
    game["index"] += 1
    game["deadline"] = time.time() + TIME_LIMIT
    game["finished_question"] = False
    game["drawing"] = None
    game["ai_answer"] = ""
    game["ai_error"] = ""
    game["timed_out"] = False


# ----------------------------
# Gemini로 그림 추측하기
# ----------------------------
def guess_with_gemini(image_bytes):
    api_key = st.secrets.get("GEMINI_API_KEY", "")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY가 없습니다. "
            "Streamlit 앱의 Secrets에 API 키를 설정해 주세요."
        )

    client = genai.Client(api_key=api_key)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    prompt = """
    너는 한국어 캐치마인드 그림 맞히기 AI야.
    사용자가 그린 그림을 보고 무엇을 그렸는지 추측해 줘.

    규칙:
    1. 답변은 한국어로 해.
    2. 가장 가능성이 높은 정답을 먼저 말해.
    3. 짧게 한 문장으로 답해.
    4. 잘 모르겠으면 추측임을 밝혀.
    5. 그림에 없는 세부 사항을 지어내지 마.
    """

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=[prompt, image],
    )

    answer = getattr(response, "text", None)

    if answer:
        answer = answer.strip()

    if not answer:
        raise ValueError(
            "Gemini API가 응답했지만 텍스트 답변이 비어 있습니다. "
            "Google AI Studio의 API 상태와 사용량을 확인해 주세요."
        )

    return answer


# ----------------------------
# 그림 제출 및 결과 저장
# ----------------------------
def submit_drawing(canvas_result, timed_out=False):
    game = st.session_state.game

    if game["finished_question"]:
        return

    game["finished_question"] = True
    game["timed_out"] = timed_out
    game["ai_answer"] = "AI가 아직 답을 내지 못했습니다."
    game["ai_error"] = ""

    image_bytes = None

    try:
        if (
            canvas_result is not None
            and canvas_result.image_data is not None
        ):
            image_array = canvas_result.image_data

            if image_array.size > 0:
                image = Image.fromarray(image_array.astype("uint8"))
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()

        if not image_bytes:
            # 캔버스 결과가 비어 있어도 결과 저장이 실패하지 않도록
            blank_image = Image.new("RGB", (800, 500), "white")
            buffer = io.BytesIO()
            blank_image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            game["ai_answer"] = "그림이 감지되지 않았습니다."

        game["drawing"] = image_bytes

        try:
            game["ai_answer"] = guess_with_gemini(image_bytes)
        except Exception as e:
            game["ai_answer"] = "AI 판별에 실패했습니다."
            game["ai_error"] = f"{type(e).__name__}: {e}"

    except Exception as e:
        game["ai_answer"] = "그림 처리 중 오류가 발생했습니다."
        game["ai_error"] = f"{type(e).__name__}: {e}"

    # 어떤 상황에서도 AI 답변이 빈 문자열로 저장되지 않게 함
    if not game.get("ai_answer", "").strip():
        game["ai_answer"] = "AI 답변이 비어 있습니다."
        if not game["ai_error"]:
            game["ai_error"] = "응답 텍스트가 비어 있습니다."

    result = {
        "number": game["index"] + 1,
        "category": game["category"],
        "keyword": game["questions"][game["index"]],
        "drawing": game["drawing"],
        "ai_answer": game["ai_answer"],
        "ai_error": game["ai_error"],
        "timed_out": timed_out,
    }

    # 같은 문제의 결과가 중복 저장되지 않도록 처리
    if not any(
        r["number"] == result["number"] for r in game["results"]
    ):
        game["results"].append(result)


# ----------------------------
# 결과 화면
# ----------------------------
def show_results():
    game = st.session_state.game

    st.title("🏁 게임 결과")
    st.write(f"카테고리: **{game['category']}**")
    st.write(f"총 {len(game['results'])}문제")

    correct_count = 0

    for result in game["results"]:
        with st.container(border=True):
            st.subheader(f"문제 {result['number']}")

            if result["drawing"]:
                st.image(
                    Image.open(io.BytesIO(result["drawing"])),
                    caption="내가 그린 그림",
                    use_container_width=True,
                )

            st.write(f"🤖 **AI의 답:** {result['ai_answer']}")
            st.write(f"✅ **실제 정답:** {result['keyword']}")

            if result["timed_out"]:
                st.caption("제한 시간이 끝나 자동 제출되었습니다.")

            if result["ai_error"]:
                with st.expander("AI 오류 내용 확인"):
                    st.code(result["ai_error"])

            if (
                result["ai_answer"]
                and result["keyword"] in result["ai_answer"]
            ):
                correct_count += 1

    st.divider()
    st.subheader(f"AI가 정답을 포함한 문제: {correct_count}/{len(game['results'])}")

    if st.button("🔄 새 게임 시작", type="primary", use_container_width=True):
        del st.session_state["game"]
        st.rerun()


# ----------------------------
# 메인 화면
# ----------------------------
st.title("🎨 AI 캐치마인드")
st.caption("그림을 그리면 AI가 정답을 추측해요!")

try:
    keyword_df = load_keywords()
except Exception as e:
    st.error("단어 파일을 읽지 못했습니다.")
    st.code(f"{type(e).__name__}: {e}")
    st.stop()

if "game" not in st.session_state:
    st.markdown("### 게임 설정")
    selected_category = st.selectbox(
        "카테고리를 선택하세요",
        CATEGORIES,
    )

    if st.button("🚀 게임 시작", type="primary", use_container_width=True):
        init_game(selected_category)
        st.rerun()

    st.info("한 게임은 5문제이며, 문제마다 60초가 주어집니다.")
    st.stop()


game = st.session_state.game

if game["index"] >= QUESTION_COUNT:
    show_results()
    st.stop()


# 문제 진행 중
question_number = game["index"] + 1
answer_keyword = game["questions"][game["index"]]

st.progress(
    question_number / QUESTION_COUNT,
    text=f"문제 {question_number} / {QUESTION_COUNT}",
)

st.write(f"카테고리: **{game['category']}**")

if not game["finished_question"]:
    remaining = max(0, int(game["deadline"] - time.time()))

    # 1초마다 화면 갱신해 남은 시간 표시
    st_autorefresh(
        interval=1000,
        key=f"timer_refresh_{question_number}",
    )

    st.subheader(f"⏱️ 남은 시간: {remaining}초")

    if remaining <= 0:
        st.warning("시간이 끝났습니다. 그림을 자동 제출합니다.")
        submit_drawing(None, timed_out=True)
        st.rerun()

    st.markdown("### 아래 캔버스에 그림을 그려 주세요")

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.0)",
        stroke_width=5,
        stroke_color="#222222",
        background_color="#FFFFFF",
        height=420,
        width=800,
        drawing_mode="freedraw",
        key=f"canvas_{question_number}",
        display_toolbar=True,
        update_streamlit=True,
    )

    if st.button(
        "✅ 그림 제출하기",
        type="primary",
        use_container_width=True,
    ):
        submit_drawing(canvas_result, timed_out=False)
        st.rerun()

else:
    st.success("제출 완료! 그림판이 잠겼습니다.")

    if game.get("drawing"):
        st.image(
            Image.open(io.BytesIO(game["drawing"])),
            caption="제출한 그림",
            use_container_width=True,
        )

    st.subheader("🤖 AI의 답")

    answer = game.get("ai_answer", "").strip()
    if answer:
        st.write(answer)
    else:
        st.error("AI 답변이 비어 있습니다.")

    if game.get("ai_error"):
        st.warning("AI 처리 중 오류가 발생했습니다. 아래 오류 내용을 확인해 주세요.")
        st.code(game["ai_error"])

    st.write(f"✅ 실제 정답: **{answer_keyword}**")

    if game["timed_out"]:
        st.info("시간 초과로 자동 제출된 문제입니다.")

    if st.button(
        "다음 문제 ➡️" if question_number < QUESTION_COUNT else "🏁 결과 확인하기",
        type="primary",
        use_container_width=True,
    ):
        reset_question()
        st.rerun()
