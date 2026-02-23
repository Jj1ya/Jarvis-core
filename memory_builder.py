import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# 1. 영구 저장소(DB) 경로 설정
DB_DIR = "./jarvis_memory"

def build_memory(file_path: str):
    print(f"[System] '{file_path}' 파일을 Jarvis의 장기 기억(Vector DB)으로 이식합니다...")

    try:
        # 2. 데이터 로드 (Ingestion)
        # 나중에는 C언어 소스코드(.c)나 PDF를 읽는 로더로 쉽게 교체할 수 있습니다.
        expanded_path = os.path.expanduser(file_path)
        loader = TextLoader(expanded_path, encoding='utf-8')
        documents = loader.load()

        # 3. 텍스트 분할 (Chunking)
        # 거대한 문서를 LLM이 소화할 수 있는 크기로 잘게 쪼갭니다.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # 500자 단위로 자름
            chunk_overlap=50 # 문맥이 끊기지 않게 50자씩 겹치게 자름
        )
        chunks = text_splitter.split_documents(documents)
        print(f"[System] 문서를 {len(chunks)}개의 데이터 조각(Chunk)으로 분할했습니다.")

        # 4. 임베딩 및 DB 저장 (Embedding & Storage)
        # nomic-embed-text 모델을 이용해 조각들을 숫자로 바꾸고 디스크에 저장합니다.
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_DIR
        )
        print(f"[System] ⚡️ 학습 완료! 데이터가 '{DB_DIR}' 폴더에 성공적으로 영구 저장되었습니다.")
        
    except Exception as e:
        print(f"[Error] 데이터 주입 중 오류 발생: {e}")

if __name__ == "__main__":
    # 테스트용 파일 생성 로직 (처음 실행 시 연습용 문서를 하나 만듭니다)
    test_file = "jarvis_rules.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("Jarvis 핵심 수칙 1: 절대 사용자의 허가 없이 로컬 데이터베이스를 삭제하지 마라.\n")
        f.write("Jarvis 핵심 수칙 2: smart-freight-ai 프로젝트의 모든 배차 알고리즘은 시간 복잡도 O(N log N)을 지향해야 한다.\n")
        f.write("Jarvis 핵심 수칙 3: C 언어에서 포인터를 사용할 때는 반드시 메모리 누수(Memory Leak)를 검증하라.\n")
        f.write("사용자의 차량은 2014년식 Audi Q5 2.0T Quattro이며, 주행거리는 약 87,000 마일이다. 엔진 오류 코드 P2293 이력이 있다.\n")
    
    # DB에 테스트 파일 주입
    build_memory(test_file)