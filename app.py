import os
from dotenv import load_dotenv


from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_huggingface import HuggingFaceEmbeddings     
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

load_dotenv()  # make sure GOOGLE_API_KEY is in your .env file

#Load the PDF
loader = PyPDFLoader("your_pdf.pdf")
docs = loader.load()

#Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
)
chunks = text_splitter.split_documents(docs)

#Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#Vector store
vectorstore = Chroma.from_documents(chunks, embeddings)

# LLM — use a supported model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4})
)

#Ask question — use invoke() instead of run()
question = "what does mab say and what is the ucb policy?"
answer = qa_chain.invoke({"query": question})
print(answer["result"])
