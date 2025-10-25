import streamlit as st
from dotenv import load_dotenv
import os
import openai
import time
import json
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from streamlit_cookies_manager import EncryptedCookieManager

# Lấy API key và password
load_dotenv()
openai.api_key = os.environ.get('OPENAI_API_KEY')
COOKIE_PASSWORD = os.environ.get("COOKIE_PASSWORD")

if not openai.api_key or not COOKIE_PASSWORD:
    st.error("Chưa tìm thấy OPENAI_API_KEY hoặc COOKIE_PASSWORD! Vui lòng thêm vào tệp .env")
    st.stop()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Bạn là một Trợ lý Tra cứu Pháp lý khách quan.
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng dựa DUY NHẤT vào các điều luật từ {law_name} được cung cấp trong phần 'CĂN CỨ PHÁP LÝ'.
Bạn phải tuân thủ các hướng dẫn sau:
1.  Đọc kỹ câu hỏi và các điều luật.
2.  Câu trả lời phải chính xác, đầy đủ, khách quan, và chi tiết.
3.  Hãy bắt đầu câu trả lời bằng cách trích dẫn điều khoản (ví dụ: "Theo Điều [số] của {law_name}...").
4.  Tuyệt đối KHÔNG thêm thông tin nào bên ngoài 'CĂN CỨ PHÁP LÝ'.
5.  Nếu không tìm thấy thông tin, hãy trả lời: "Dựa trên các thông tin được cung cấp, tôi không tìm thấy điều luật quy định về vấn đề này."


CĂN CỨ PHÁP LÝ (từ {law_name}):
{context}
---
CÂU HỎI: {question}
TRẢ LỜI (dựa vào {law_name}):
"""


def get_law_name_from_source(source_path: str) -> str:
    filename = os.path.basename(source_path)
    if "Bo_Luat_Hinh_Su_Hop_Nhat" in filename:
        return "Bộ luật Hình sự 2025"
    elif "luat_dan_su" in filename:
        return "Bộ luật Dân sự 2015"
    elif "luat_dat_dai" in filename:
        return "Bộ luật Đất đai 2024"
    return "Văn bản pháp lý được cung cấp"


@st.cache_resource
def load_chroma_db():
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    return db


@st.cache_resource
def load_chat_model():
    model = ChatOpenAI(model="gpt-3.5-turbo")
    return model


db = load_chroma_db()
model = load_chat_model()


def get_rag_response(query_text: str):
    results = db.similarity_search_with_relevance_scores(query_text, k=5)

    if not results or results[0][1] < 0.7:
        return "Dựa trên các thông tin được cung cấp, tôi không tìm thấy điều luật quy định về vấn đề này.", []

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    source_file_path = results[0][0].metadata.get("source", "")
    law_name = get_law_name_from_source(source_file_path)

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(
        context=context_text,
        question=query_text,
        law_name=law_name
    )
    # Fetch cho ChatGPT API
    response_ai_message = model.invoke(prompt)
    response_text = response_ai_message.content

    sources = [get_law_name_from_source(doc.metadata.get("source", "")) for doc, _score in results]
    unique_sources = list(set(sources))

    return response_text, unique_sources


# Config trang

st.set_page_config(page_title="Chatbot Pháp lý", page_icon="⚖️", layout="wide")

# Khởi tạo trình quản lý cookie
cookies = EncryptedCookieManager(
    password=COOKIE_PASSWORD,
    prefix="luat_chatbot/cookie/",
)
if not cookies.ready:
    st.stop()


# Tải tất cả các cuộc hội thoại từ cookie vào session_state khi bắt đầu
if "chat_sessions" not in st.session_state:
    cookie_data = cookies.get("chat_sessions")
    if cookie_data:
        try:
            st.session_state.chat_sessions = json.loads(cookie_data)
        except json.JSONDecodeError:
            # Nếu cookie bị hỏng, bắt đầu lại với dict rỗng
            st.session_state.chat_sessions = {}
    else:
        # Nếu không có cookie, bắt đầu với dict rỗng
        st.session_state.chat_sessions = {}

# Xác định cuộc hội thoại nào đang hoạt động
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None


# Hàm lưu cookie
def save_sessions_to_cookie():
    # Convert dict to JSON
    json_data = json.dumps(st.session_state.chat_sessions)
    cookies["chat_sessions"] = json_data


# UI Sidebar

with st.sidebar:
    st.title("⚖️ Chatbot Luật")


    if st.button("Tạo cuộc trò chuyện mới", use_container_width=True):
        st.session_state.active_chat_id = None
        st.rerun()

    st.divider()

    # Hiển thị lịch sử truy vấn
    st.subheader("Lịch sử truy vấn")
    sorted_chat_ids = sorted(st.session_state.chat_sessions.keys(), reverse=True)

    for chat_id in sorted_chat_ids:
        chat_data = st.session_state.chat_sessions[chat_id]


        if st.button(chat_data.get("title", "Cuộc trò chuyện không tên"), use_container_width=True, type="secondary"):
            st.session_state.active_chat_id = chat_id
            st.rerun()


    st.divider()
    if st.button("🗑️ Xóa lịch sử", use_container_width=True, type="secondary"):
        st.session_state.chat_sessions = {}
        st.session_state.active_chat_id = None
        # [SỬA] Sử dụng 'del' để xóa cookie, thay vì lưu đè
        if "chat_sessions" in cookies:
            del cookies["chat_sessions"]
        st.rerun()



# Kiểm tra xem có cuộc hội thoại nào đang hoạt động không
active_id = st.session_state.active_chat_id

if active_id is None:
    # Đây là "Trang trắng" khi mới vào hoặc nhấn "Tạo mới"
    st.info("Bắt đầu cuộc trò chuyện mới bằng cách đặt câu hỏi của bạn bên dưới.")

else:

    # Lấy lịch sử tin nhắn của cuộc hội thoại đang hoạt động
    messages = st.session_state.chat_sessions[active_id].get("messages", [])

    # Show các tin nhắn đã có
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# Xử lý input

if prompt := st.chat_input("Bạn muốn hỏi gì về luật?"):
    if active_id is None:
        # Tạo một ID mới
        active_id = f"chat_{int(time.time())}"
        # Tạo tiêu đề (lấy 40 ký tự đầu của câu hỏi)
        new_title = prompt[:40] + "..."
        # Tạo cuộc hội thoại mới
        st.session_state.chat_sessions[active_id] = {
            "title": new_title,
            "messages": []
        }
        # Đặt nó làm cuộc hội thoại hoạt động
        st.session_state.active_chat_id = active_id

    # thêm câu hỏi vào lịch sử
    st.session_state.chat_sessions[active_id]["messages"].append(
        {"role": "user", "content": prompt}
    )

    # Hiển thị câu hỏi của người dùng ngay lập tức
    with st.chat_message("user"):
        st.markdown(prompt)

    # Lấy câu trả lời từ RAG
    with st.chat_message("assistant"):
        with st.spinner("Đang tra cứu và tổng hợp..."):
            response, sources = get_rag_response(prompt)

            if sources:
                response_with_sources = f"{response}\n\n*Nguồn tham khảo: {', '.join(sources)}*"
            else:
                response_with_sources = response

            st.markdown(response_with_sources)

    # Thêm câu trả lời của AI vào lịch sử
    st.session_state.chat_sessions[active_id]["messages"].append(
        {"role": "assistant", "content": response_with_sources}
    )

    # Lưu tất cả thay đổi vào cookie và tải lại trang
    save_sessions_to_cookie()
    st.rerun()