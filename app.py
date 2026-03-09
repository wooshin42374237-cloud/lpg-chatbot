import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# --- 설정 (Gemini API Key 입력 필요) ---
# 🚨 주의: 이곳에 새로 발급받은 구글 API 키를 넣어주세요!
os.environ["GOOGLE_API_KEY"] = "AIzaSyDDEvdAhaC6YAEYVaO3VstozAZdCdd1lHc"

st.set_page_config(page_title="인허가 문서 AI 챗봇", layout="wide")
st.title("LPG 터미널 인허가 & 규격 검토 챗봇 🤖")

# --- PDF 텍스트 추출 함수 ---
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted
    return text

# --- 텍스트를 벡터 DB에 저장 함수 ---
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index") # 로컬 폴더에 DB 저장

# --- 최신 방식(LCEL) 챗봇 응답 생성 함수 ---
def get_conversational_chain():
    prompt_template = """
    주어진 문맥(Context)을 바탕으로 질문에 최대한 자세히 답변해 주세요.
    만약 문맥에 없는 내용이라면 "제공된 문서에서는 해당 내용을 찾을 수 없습니다."라고 답변해 주세요.
    
    Context:\n {context}?\n
    Question: \n{question}\n
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | model | StrOutputParser()
    return chain

# --- 사이드바: 파일 업로드 영역 ---
with st.sidebar:
    st.header("문서 업로드 (DB 저장)")
    pdf_docs = st.file_uploader("관련 인허가 서류(PDF)를 업로드하세요.", accept_multiple_files=True)
    if st.button("문서 처리 및 DB 저장"):
        if not pdf_docs:
            st.warning("⚠️ 파일을 먼저 업로드해주세요!")
        else:
            with st.spinner("문서를 분석하고 데이터베이스에 저장 중입니다..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("🚨 에러: PDF에서 텍스트를 읽을 수 없습니다.")
                    else:
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        text_chunks = text_splitter.split_text(raw_text)
                        if not text_chunks:
                            st.error("🚨 에러: 텍스트를 분할하는 데 실패했습니다.")
                        else:
                            get_vector_store(text_chunks)
                            st.success("데이터베이스 저장이 완료되었습니다!")
                except Exception as e:
                    # 구글 API 통신 에러가 나면 무한 로딩 대신 화면에 에러를 즉시 출력합니다.
                    st.error(f"🚨 구글 API 통신 에러가 발생했습니다:\n\n{e}")

# --- 메인 화면: 챗봇 UI ---
user_question = st.text_input("업로드된 문서에 대해 질문해 주세요 (예: API 625에서 롤오버(Rollover)를 어떻게 방지하라고 되어 있나요?)")

if user_question:
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        
        # 최신 방식에 맞게 검색된 문서를 텍스트로 결합
        context = "\n\n".join([doc.page_content for doc in docs])
        
        chain = get_conversational_chain()
        response = chain.invoke({"context": context, "question": user_question})
        
        st.write("### 💡 AI 답변:")
        st.info(response)
    except Exception as e:
        st.warning(f"먼저 좌측 사이드바에서 문서를 업로드하고 DB 저장 버튼을 눌러주세요. (상세 에러: {e})")
