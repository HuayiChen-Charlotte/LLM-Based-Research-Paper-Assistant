import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_documents(folder):
    documents = []
    if not os.path.exists(folder):
        print(f"Folder '{folder}' not found.")
        return []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            print(f"Loading {file}")
            loader = PyPDFLoader(path)
            # load_and_split can sometimes handle layout better than load()
            documents.extend(loader.load())
    return documents

def split_documents(documents):
    # Larger chunks capture full technical sections like the Abstract
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks

def build_vector_db(papers_folder):
    # Free local embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if os.path.exists("faiss_index"):
        print("Loading existing vector database... (Delete 'faiss_index' folder to refresh)")
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    print("Building new vector database...")
    docs = load_documents(papers_folder)
    if not docs:
        print("No PDFs found in 'papers' folder.")
        return None
        
    chunks = split_documents(docs)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("faiss_index")
    return db