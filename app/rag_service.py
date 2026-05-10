import os
import re
import shutil
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PDF_FOLDER = str(BASE_DIR / "data" / "pdfs")
DB_PATH = os.getenv("CHROMA_DB_PATH", "vector_db")


def list_uploaded_pdfs():
    os.makedirs(PDF_FOLDER, exist_ok=True)
    return sorted([
        file_name
        for file_name in os.listdir(PDF_FOLDER)
        if file_name.lower().endswith(".pdf")
    ])


def load_pdfs():
    all_documents = []
    os.makedirs(PDF_FOLDER, exist_ok=True)

    for file_name in list_uploaded_pdfs():
        file_path = os.path.join(PDF_FOLDER, file_name)
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        all_documents.extend(documents)

    return all_documents


def create_vector_db():
    documents = load_pdfs()

    if not documents:
        if os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
        return "No PDF files found. Please upload PDF files first."

    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    db.persist()

    return f"{len(list_uploaded_pdfs())} PDF file(s) indexed successfully."


def load_qa_chain():
    embeddings = OpenAIEmbeddings()

    db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    retriever = db.as_retriever(search_kwargs={"k": 5})

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    return qa_chain


def count_word_in_pdfs(word: str):
    documents = load_pdfs()

    total_count = 0
    file_counts = {}

    pattern = r"\b" + re.escape(word.lower()) + r"\b"

    for doc in documents:
        source = Path(doc.metadata.get("source", "unknown")).name
        text = doc.page_content.lower()

        count = len(re.findall(pattern, text))

        file_counts[source] = file_counts.get(source, 0) + count
        total_count += count

    return {
        "word": word,
        "total_count": total_count,
        "file_counts": file_counts
    }


def detect_count_question(question: str):
    question_lower = question.lower()

    count_keywords = [
        "how many times",
        "count",
        "number of times",
        "mentioned",
        "appear",
        "appears",
        "occurrence",
        "occurrences"
    ]

    if not any(keyword in question_lower for keyword in count_keywords):
        return None

    patterns = [
        r"how many times is ['\"]?(.+?)['\"]? mentioned",
        r"how many times ['\"]?(.+?)['\"]? is mentioned",
        r"count ['\"]?(.+?)['\"]?",
        r"number of times ['\"]?(.+?)['\"]?",
        r"how many times ['\"]?(.+?)['\"]? appears",
        r"how many times ['\"]?(.+?)['\"]? appear"
    ]

    for pattern in patterns:
        match = re.search(pattern, question_lower)
        if match:
            word = match.group(1).strip()

            remove_words = [
                "in both pdfs",
                "in the pdfs",
                "in pdfs",
                "in both documents",
                "in the documents",
                "in documents",
                "in both files",
                "in the files"
            ]

            for remove_word in remove_words:
                word = word.replace(remove_word, "")

            word = word.strip(" ?.,'\"")

            if word:
                return word

    return None


def smart_answer(question: str, qa_chain):
    count_word = detect_count_question(question)

    if count_word:
        result = count_word_in_pdfs(count_word)

        file_details = ""
        for file_name, count in result["file_counts"].items():
            file_details += f"\n- {file_name}: {count}"

        return (
            f'The word "{result["word"]}" is mentioned '
            f'{result["total_count"]} times in total across all uploaded PDFs.'
            f"\n\nBreakdown:{file_details}"
        )

    answer = qa_chain.run(question)
    return answer.strip() if isinstance(answer, str) else answer
