import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()



class ResumeQA:
    
    def __init__(self, pdf_path: str, index_path: str = "resume_faiss_index"):
        self.pdf_path = pdf_path
        self.index_path = index_path
        self.api_key = os.getenv("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError("⚠️ GEMINI_API_KEY not found. Please set it in .env")

        # Load & process resume
        loader = PyPDFLoader(pdf_path)
        data = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(data)

        # Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=self.api_key
        )

        # Vector DB (FAISS)
        if not os.path.exists(index_path):
            vector_index = FAISS.from_documents(docs, embeddings)
            vector_index.save_local(index_path)
        else:
            vector_index = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

        # Gemini LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=self.api_key,
            temperature=0.7
        )

        # Prompt
        system_prompt = """
        You are a professional AI assistant that answers questions based only on the resume of Nachiket Shinde.
        Your goal is to help recruiters, HR professionals, and collaborators quickly understand his skills, projects,
        certifications, and achievements.

        Guidelines:
        - Always be concise, clear, and professional.
        - Highlight Nachiket’s strengths and technical expertise.
        - If a question is unrelated to the resume, politely respond with:
          "I can only answer questions related to Nachiket Shinde's resume."

        Resume context:
        {context}

        Question: {question}
        Answer:
        """

        prompt = PromptTemplate(
            template=system_prompt,
            input_variables=["context", "question"]
        )

        # QA Chain
        self.chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_index.as_retriever(),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt}
        )

    def ask(self, question: str) -> str:
        return self.chain.run(question)
