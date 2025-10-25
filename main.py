
from langchain_community.document_loaders import DirectoryLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai
from dotenv import load_dotenv
import os
import shutil


load_dotenv()

openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = "chroma"
DATA_PATH = "data/laws"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


# def save_to_chroma(chunks: list[Document]):
#     # Clear out the database first.
#     if os.path.exists(CHROMA_PATH):
#         shutil.rmtree(CHROMA_PATH)
#
#     # Create a new DB from the documents.
#     db = Chroma.from_documents(
#         chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
#     )
#     db.persist()
#     print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
def save_to_chroma(chunks: list[Document]):
    # Xóa DB cũ đi
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # 1. Xác định một kích thước lô (batch size) an toàn
    # 1474 chunks đã thất bại, chúng ta sẽ thử chia nhỏ hơn, ví dụ: 200 chunks mỗi lần.
    batch_size = 200

    # 2. Tạo cơ sở dữ liệu Chroma với lô *đầu tiên*
    print(f"Đang tạo cơ sở dữ liệu với lô đầu tiên (tối đa {batch_size} chunks)...")

    # Lấy lô đầu tiên
    first_batch = chunks[:batch_size]

    # Tạo DB mới với lô đầu tiên
    db = Chroma.from_documents(
        first_batch,
        OpenAIEmbeddings(),
        persist_directory=CHROMA_PATH
    )

    # 3. Thêm các lô *còn lại* vào DB đã tồn tại
    total_batches = (len(chunks) // batch_size) + 1

    # Vòng lặp bắt đầu từ vị trí batch_size (vì lô đầu tiên đã xong)
    for i in range(batch_size, len(chunks), batch_size):
        # Lấy lô tiếp theo
        batch = chunks[i:i + batch_size]

        current_batch_num = (i // batch_size) + 1
        print(f"Đang thêm lô {current_batch_num}/{total_batches} ({len(batch)} chunks)...")

        # Sử dụng .add_documents() để thêm vào DB đã có
        db.add_documents(batch)

    # 4. Lưu (mặc dù không cần thiết với Chroma mới, nhưng không gây hại)
    db.persist()
    print(f"Đã lưu thành công {len(chunks)} chunks vào {CHROMA_PATH}.")


if __name__ == "__main__":
    main()