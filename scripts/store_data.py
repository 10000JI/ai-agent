# Store PDF documents in Elasticsearch using OpenAI embeddings
# Reference: https://github.com/ezimuel/langchain-ollama-elasticsearch

import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import pymupdf4llm
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from elasticsearch import Elasticsearch


# ============================================================
# 설정
# ============================================================
INDEX_NAME = os.getenv("ES_INDEX", "edu-cosmetic")
PDF_DIR = Path(__file__).parent.parent / "docs" / "pdfs"

# Elasticsearch 클라이언트
es_client = Elasticsearch(
    os.getenv("ES_URL"),
    basic_auth=(os.getenv("ES_USERNAME"), os.getenv("ES_PASSWORD")),
    verify_certs=False,
)

# OpenAI 임베딩
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# ElasticsearchStore
vector_store = ElasticsearchStore(
    index_name=INDEX_NAME,
    embedding=embeddings,
    client=es_client,
)

# 청크 분할 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "],
)


# ============================================================
# Step 1: ES 연결 및 인덱스 확인
# ============================================================
def check_elasticsearch():
    """ES 연결 확인 및 인덱스 중복 체크"""
    if not es_client.ping():
        print("Elasticsearch 연결 실패. .env 파일의 ES 설정을 확인하세요.")
        sys.exit(1)
    print("Elasticsearch 연결 성공!")

    if es_client.indices.exists(index=INDEX_NAME):
        print(f"\n인덱스 '{INDEX_NAME}'이 이미 존재합니다.")
        answer = input("기존 인덱스를 삭제하고 새로 생성할까요? (y/N): ").strip().lower()
        if answer == "y":
            es_client.indices.delete(index=INDEX_NAME)
            print(f"인덱스 '{INDEX_NAME}' 삭제 완료.")
        else:
            print("인덱싱을 중단합니다.")
            sys.exit(0)


# ============================================================
# Step 2: PDF 파일 로드 및 청크 분할
# ============================================================
def load_and_split_pdfs() -> list[list[Document]]:
    """PDF 파일들을 읽고 청크로 분할"""
    if not PDF_DIR.exists():
        print(f"PDF 디렉토리가 없습니다: {PDF_DIR}")
        sys.exit(1)

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"PDF 파일이 없습니다: {PDF_DIR}")
        sys.exit(1)

    print(f"\n발견된 PDF 파일: {len(pdf_files)}개")

    all_splits = []
    for pdf_path in pdf_files:
        print(f"\n{'='*60}")
        print(f"처리 중: {pdf_path.name}")

        # pymupdf4llm으로 2단 컬럼 레이아웃까지 올바르게 파싱
        md_text = pymupdf4llm.to_markdown(str(pdf_path), pages=None)
        doc = Document(
            page_content=md_text,
            metadata={"source": pdf_path.name},
        )

        total_chars = len(md_text)
        print(f"  추출 텍스트: {total_chars:,}자")

        chunks = text_splitter.split_documents([doc])
        print(f"  청크 수: {len(chunks)}개")

        all_splits.append(chunks)

    return all_splits


# ============================================================
# Step 3: Elasticsearch에 임베딩 + 저장
# ============================================================
def store_to_elasticsearch(all_splits: list[list[Document]]):
    """청크들을 임베딩하여 ES에 저장"""
    print(f"\n{'='*60}")
    print("Elasticsearch에 저장 중...")

    total_chunks = 0
    for chunks in all_splits:
        ElasticsearchStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            client=es_client,
            index_name=INDEX_NAME,
        )
        total_chunks += len(chunks)
        print(f"  저장 완료: {len(chunks)}개 청크 → {INDEX_NAME}")

    return total_chunks


# ============================================================
# Main
# ============================================================
def main():
    print(f"인덱스: {INDEX_NAME}")
    print(f"PDF 경로: {PDF_DIR}")

    # Step 1: ES 연결 및 인덱스 확인
    check_elasticsearch()

    # Step 2: PDF 파일 로드 및 청크 분할
    all_splits = load_and_split_pdfs()

    # Step 3: ES에 임베딩 + 저장
    total_chunks = store_to_elasticsearch(all_splits)

    # 결과 출력
    print(f"\n{'='*60}")
    print(f"완료! {len(all_splits)}개 PDF → {total_chunks:,}개 청크 인덱싱")
    print(f"인덱스: {INDEX_NAME}")

    # 인덱스 문서 수 확인
    count = es_client.count(index=INDEX_NAME)["count"]
    print(f"ES 인덱스 문서 수: {count:,}개")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
