import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'mps'},
            encode_kwargs={'normalize_embeddings': True}
        )



if __name__ == "__main__":

    path = "/Users/igor.varha/Zotero/storage"

    # Using BGE-large model which is optimized for academic text
    embeddings = Embedder()


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Increased from 500 to 1500 for better context
        chunk_overlap=250,  # Increased from 50 to 250 for better continuity
        length_function=len,
        is_separator_regex=False
    )
    chroma = Chroma(persist_directory="chroma_db/", embedding_function=embeddings.embeddings)

    # Batch sizes
    pdf_batch_size = 100  # Number of PDFs to process at once
    doc_batch_size = 1000  # Number of document chunks to add to ChromaDB at once

    # Get all PDF files from the storage directory
    pdf_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))

    # Process PDFs in batches
    for i in range(0, len(pdf_files), pdf_batch_size):
        batch = pdf_files[i:i + pdf_batch_size]
        
        # Load and split documents
        documents = []
        for pdf_path in batch:
            try:
                # Load PDF text
                loader = PyPDFLoader(pdf_path)
                text = loader.load()
                
                # Split into chunks
                chunks = splitter.split_documents(text)
                documents.extend(chunks)
            except Exception as e:
                print(f"Error processing {pdf_path}: {str(e)}")
                continue
        
        # Add documents to vector store in smaller batches
        if documents:
            for j in range(0, len(documents), doc_batch_size):
                doc_batch = documents[j:j + doc_batch_size]
                chroma.add_documents(doc_batch)
                print(f"Added batch of {len(doc_batch)} documents to vector store")
            
            print(f"Completed processing batch of {len(batch)} PDFs")

    # Persist the vector store
    chroma.persist()
    print("Vector store creation completed")


