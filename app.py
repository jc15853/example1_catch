
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


# ==================================================
# 기본 설정
# ==================================================
st.set_page_config(
    page_title="AI 캐치마인드",
    page_icon="🎨",
    layout="wide",
)

CSV_PATH = Path(__file__).parent / "keyword.csv"
QUESTION_COUNT = 5
TIME_LIMIT = 60
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 420

CATEGORIES = ["동물", "과일", "채소", "사물", "교통수단"]


# ==================================================
# 단어 데이터 불러오기
# ==================================================
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
            "keyword.csv 파일에는 category, keyword 열이 필요합니다."
        )

    df = df.dropna(subset=["category", "keyword"]).copy()
    df["category"] = df["category"].astype(str).str.strip()
    df["keyword"] = df["keyword"].astype(str).str.strip()

    df = df[
        (df["category"] != "")
        & (df["keyword"] != "")
    ]

    return df


# ==================================================
# 게임 시작
# ==================================================
def init_game(category):
    df = load_keywords()

    available_keywords = (
        df.loc[df["category"] == category, "keyword"]
        .drop_duplicates()
        .tolist()
    )

    if len(available_keywords) < QUESTION_COUNT:
        st.error(
            f"{category} 카테고리의 단어가 부족합니다. "
            f"keyword.csv에 서로 다른 단어를 최소 "
            f"{QUESTION_COUNT}개 넣어 주세요."
        )
        return

    questions = random.sample(
        available_keywords,
        QUESTION_COUNT,
    )

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


# ==================================================
# Gemini AI로 그림 추측
# ==================================================
def guess_with_gemini(image_bytes):
    api_key = st.secrets.get("GEMINI_API_KEY", "")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY가 설정되지 않았습니다. "
            "Streamlit 앱의 Settings > Secrets에서 API 키를 설정하세요."
        )

    client = genai.Client(api_key=api_key)

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    prompt = """
너는 한국어 캐치마인드 그림 맞히기 전문가야.

사용자가 그린 그림을 보고 무엇을 그렸는지 추측해.

규칙:
1. 그림에서 보이는 모양과 특징을 자세히 살펴봐.
2. 가장 가능성이 높은 대상 하나를 추측해.
3. 반드시 한국어로 답해.
4. 가능하면 구체적인 이름을 말해.
5. 확신이 낮더라도 가장 가능성 높은 답을 제시해.
6. "모르겠음"이라고만 답하지 마.
7. 그림이 불분명하다면 "추측:"을 붙여 가능성 높은 답을 말해.
8. 답변은 짧게 작성해.

예시:
추측: 늑대
"""

    # Gemini API 호출
    # 모델 사용 가능 여부는 Google AI Studio에서 확인하세요.
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=[prompt, image],
    )

    answer = getattr(response, "text", None)

    if answer:
        answer = answer.strip()

    if not answer:
        raise ValueError(
            "Gemini가 빈 텍스트를 반환했습니다. "
            "모델 지원 여부와 API 응답을 확인하세요."
        )

    return answer


# ==================================================
# 그림 제출 및 결과 저장
# ==================================================
def submit_drawing(canvas_result=None, timed_out=False):
    game = st.session_state.game

    # 이미 제출된 문제는 중복 제출하지 않음
    if game.get("finished_question", False):
        return

    # 먼저 잠금 처리
    game["finished_question"] = True
    game["timed_out"] = bool(timed_out)
    game["ai_answer"] = "AI 답변을 처리하지 못했습니다."
    game["ai_error"] = ""

    image_bytes = None

    try:
        if (
            canvas_result is not None
            and canvas_result.image_data is not None
        ):
            image_array = canvas_result.image_data

            if image_array.size > 0:
                image = Image.fromarray(
                    image_array.astype("uint8")
                )

                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()

        # 캔버스 데이터가 없으면 흰 배경 이미지 생성
        if not image_bytes:
            blank_image = Image.new(
                "RGB",
                (CANVAS_WIDTH, CANVAS_HEIGHT),
                "white",
            )

            buffer = io.BytesIO()
            blank_image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()

        game["drawing"] = image_bytes

        # AI 추측
        try:
            game["ai_answer"] = guess_with_gemini(
                image_bytes
            )
        except Exception as e:
            game["ai_answer"] = "AI 판별에 실패했습니다."
            game["ai_error"] = (
                f"{type(e).__name__}: {str(e)}"
            )

    except Exception as e:
        game["ai_answer"] = "그림 처리 중 오류가 발생했습니다."
        game["ai_error"] = (
            f"{type(e).__name__}: {str(e)}"
        )

    # AI 답변이 비어 있는 상황 방지
    if not str(game.get("ai_answer", "")).strip():
        game["ai_answer"] = "AI 답변이 비어 있습니다."

        if not game.get("ai_error"):
            game["ai_error"] = (
                "AI 응답에 표시할 텍스트가 없습니다."
            )

    # 실제 정답
    index = game["index"]
    keyword = game["questions"][index]

    result = {
        "number": index + 1,
        "category": game["category"],
        "keyword": keyword,
        "drawing": game["drawing"],
        "ai_answer": game["ai_answer"],
        "ai_error": game["ai_error"],
        "timed_out": bool(timed_out),
    }

    # 문제별 결과 중복 저장 방지
    already_saved = any(
        item.get("number") == index + 1
        for item in game["results"]
    )

    if not already_saved:
        game["results"].append(result)


# ==================================================
# 다음 문제로 이동
# ==================================================
def advance_question():
    game = st.session_state.game

    game["index"] += 1
    game["finished_question"] = False
    game["drawing"] = None
    game["ai_answer"] = ""
    game["ai_error"] = ""
    game["timed_out"] = False

    if game["index"] < QUESTION_COUNT:
        game["deadline"] = time.time() + TIME_LIMIT


# ==================================================
# 결과 화면
# ==================================================
def show_results():
    game = st.session_state.game
    results = game.get("results", [])

    st.title("🏁 게임 결과")
    st.write(f"카테고리: **{game['category']}**")
    st.write(f"완료한 문제: **{len(results)} / {QUESTION_COUNT}**")

    matched_count = 0

    for result in results:
        with st.container(border=True):
            st.subheader(
                f"문제 {result.get('number', '?')}"
            )

            drawing = result.get("drawing")

            if drawing:
                try:
                    st.image(
                        Image.open(io.BytesIO(drawing)),
                        caption="제출한 그림",
                        use_container_width=True,
                    )
                except Exception:
                    st.warning("그림 이미지를 표시할 수 없습니다.")

            ai_answer = result.get(
                "ai_answer",
                "AI 답변이 없습니다.",
            )

            keyword = result.get(
                "keyword",
                "정답 정보 없음",
            )

            st.markdown(f"**🤖 AI의 답:** {ai_answer}")
            st.markdown(f"**✅ 실제 정답:** {keyword}")

            if result.get("timed_out", False):
                st.caption("제한 시간 종료로 자동 제출된 문제입니다.")

            error_message = result.get("ai_error", "")

            if error_message:
                with st.expander("AI 오류 내용 확인"):
                    st.code(error_message)

            if keyword and keyword in ai_answer:
                matched_count += 1

    st.divider()

    st.subheader(
        f"AI 답변에 정답이 포함된 문제: "
        f"{matched_count} / {len(results)}"
    )

    if st.button(
        "🔄 새 게임 시작",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.pop("game", None)
        st.rerun()


# ==================================================
# 메인 화면
# ==================================================
st.title("🎨 AI 캐치마인드")
st.caption(
    "그림을 그리면 AI가 무엇인지 추측합니다. "
    "문제마다 제한 시간은 60초입니다."
)

# 데이터 파일 오류 처리
try:
    keyword_df = load_keywords()
except Exception as e:
    st.error("단어 파일을 불러오지 못했습니다.")
    st.code(f"{type(e).__name__}: {e}")
    st.stop()


# ==================================================
# 시작 화면
# ==================================================
if "game" not in st.session_state:
    st.subheader("게임 설정")

    selected_category = st.selectbox(
        "카테고리를 선택하세요",
        CATEGORIES,
    )

    category_count = len(
        keyword_df[
            keyword_df["category"] == selected_category
        ]["keyword"].drop_duplicates()
    )

    st.caption(
        f"선택한 카테고리의 단어 수: {category_count}개"
    )

    if st.button(
        "🚀 게임 시작",
        type="primary",
        use_container_width=True,
    ):
        init_game(selected_category)

        if "game" in st.session_state:
            st.rerun()

    st.info(
        "총 5문제입니다. 각 문제에서 그림을 제출하면 "
        "AI의 추측과 실제 정답을 확인할 수 있습니다."
    )

    st.stop()


# ==================================================
# 현재 게임 상태 확인
# ==================================================
game = st.session_state.game

# 구버전 세션 상태에 빠진 값이 있어도 오류가 나지 않게 보정
game.setdefault("category", CATEGORIES[0])
game.setdefault("questions", [])
game.setdefault("index", 0)
game.setdefault("results", [])
game.setdefault("deadline", time.time() + TIME_LIMIT)
game.setdefault("finished_question", False)
game.setdefault("drawing", None)
game.setdefault("ai_answer", "")
game.setdefault("ai_error", "")
game.setdefault("timed_out", False)

# 문제 데이터가 없으면 새 게임으로 돌아감
if not game["questions"]:
    st.session_state.pop("game", None)
    st.warning("게임 데이터가 없어 새 게임을 시작해 주세요.")
    st.rerun()

# 모든 문제 완료
if game["index"] >= QUESTION_COUNT:
    show_results()
    st.stop()


# ==================================================
# 문제 화면
# ==================================================
question_number = game["index"] + 1
answer_keyword = game["questions"][game["index"]]

st.progress(
    question_number / QUESTION_COUNT,
    text=f"문제 {question_number} / {QUESTION_COUNT}",
)

st.write(f"카테고리: **{game['category']}**")


# ==================================================
# 그림 그리기 화면
# ==================================================
if not game["finished_question"]:
    remaining = max(
        0,
        int(game["deadline"] - time.time()),
    )

    # 시간 표시 갱신
    st_autorefresh(
        interval=1000,
        key=f"timer_{question_number}",
    )

    st.subheader(f"⏱️ 남은 시간: {remaining}초")

    if remaining <= 0:
        st.warning("시간이 끝났습니다. 그림을 자동 제출합니다.")
        submit_drawing(
            canvas_result=None,
            timed_out=True,
        )
        st.rerun()

    st.markdown("### 🖌️ 여기에 그림을 그려 주세요")

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=5,
        stroke_color="#222222",
        background_color="#FFFFFF",
        height=CANVAS_HEIGHT,
        width=CANVAS_WIDTH,
        drawing_mode="freedraw",
        display_toolbar=True,
        update_streamlit=True,
        key=f"canvas_{question_number}",
    )

    if st.button(
        "✅ 그림 제출하기",
        type="primary",
        use_container_width=True,
    ):
        submit_drawing(
            canvas_result=canvas_result,
            timed_out=False,
        )
        st.rerun()


# ==================================================
# 제출 후 화면
# ==================================================
else:
    st.success("제출 완료! 그림판이 잠겼습니다.")

    drawing = game.get("drawing")

    if drawing:
        st.image(
            Image.open(io.BytesIO(drawing)),
            caption="제출한 그림",
            use_container_width=True,
        )

    st.subheader("🤖 AI의 답")

    ai_answer = str(
        game.get("ai_answer", "")
    ).strip()

    if ai_answer:
        st.write(ai_answer)
    else:
        st.error("AI 답변이 비어 있습니다.")

    if game.get("ai_error"):
        st.warning(
            "AI 처리 중 오류가 발생했습니다. "
            "아래 메시지를 확인해 주세요."
        )
        st.code(game["ai_error"])

    st.markdown(
        f"**✅ 실제 정답: {answer_keyword}**"
    )

    if game.get("timed_out", False):
        st.info("제한 시간이 끝나 자동 제출되었습니다.")

    if question_number < QUESTION_COUNT:
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
