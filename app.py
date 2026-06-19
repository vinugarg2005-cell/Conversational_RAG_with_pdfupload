from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader,CSVLoader,Docx2txtLoader,TextLoader,UnstructuredExcelLoader
from langchain_chroma import Chroma
import streamlit as st
import os
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableWithMessageHistory
from dotenv import load_dotenv
load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    os.environ['HF_TOKEN'] = hf_token
groq_key = os.getenv("GROQ_API_KEY")
if groq_key:
    os.environ['GROQ_API_KEY'] = groq_key

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title("Conversational RAG with pdf upload and chat history")
st.write("Enter your pdf:")

api_key = st.text_input("Enter your groq api key",type="password")
if api_key:
    llm = ChatGroq(model="llama-3.3-70b-versatile",groq_api_key=api_key)
    session_id = st.text_input("Enter your session ID",value="default_session")
    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Upload a PDF file",ype=["pdf", "csv", "xlsx", "docx", "txt"],accept_multiple_files=True)
    if uploaded_files:
        documents = []
        for uploaded_file in uploaded_files:
            temp_path = f"./temp_{uploaded_file.name}"
            with open(temp_path, "wb") as file:
                file.write(uploaded_file.getvalue())

            # File type detect karo
            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(temp_path)
            elif uploaded_file.name.endswith(".csv"):
                loader = CSVLoader(temp_path)
            elif uploaded_file.name.endswith(".xlsx"):
                loader = UnstructuredExcelLoader(temp_path)
            elif uploaded_file.name.endswith(".docx"):
                loader = Docx2txtLoader(temp_path)
            elif uploaded_file.name.endswith(".txt"):
                loader = TextLoader(temp_path)
            else:
                st.warning(f"Unsupported file type: {uploaded_file.name}")
                continue
            docs=loader.load()
            documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
        chunks = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=chunks,embedding=embeddings)
        retriever = vectorstore.as_retriever()

        contexualized_q_system_prompt = [
            "Given a chat history and a latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do not answer the question, "
            "just reformulate it if needed and otherwise return it as it is."
        ]

        contexualized_q_prompt = ChatPromptTemplate.from_messages(
            [
            ("system",contexualized_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm,retriever,contexualized_q_prompt)

        system_prompt = [
            "You are an assistant for question answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the answer concise."
            "\n\n"
            "{context}"
        ]

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm,prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)


        def get_session_history(session:str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        user_input = st.text_input("Ask your question")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input":user_input},
                config={"configurable":{"session_id":session_id}}
            )

            st.write(st.session_state.store)
            st.write("Assistant:",response["answer"])
            st.write("Chat History:",session_history.messages)

else:
    st.warning("Please enter the Groq API Key")









