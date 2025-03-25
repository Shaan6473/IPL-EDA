import streamlit as st
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


@st.cache_resource
def initialize_vectorstore(_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(_chunks, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Failed to initialize the vectorstore: {e}")
        return None


def app():
    st.markdown('''
    <h1 style='text-align:center; color: #e8630a;'><strong>🧠IPL Match Query Bot (2008-2024)🧠</strong></h1>
    <hr style="border-top: 3px solid #e8630a;">
    ''', unsafe_allow_html=True)

    # Load environment variables
    load_dotenv()
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    langchain_api_key = st.secrets["LANGCHAIN_API_KEY"]

    # Load CSV file and display DataFrame in Streamlit
    loader = CSVLoader('./matches_2008-2024.csv')
    docs = loader.load()

    # Display the dataframe
    csv_path = './matches_2008-2024.csv'
    df = pd.read_csv(csv_path)
    st.subheader("Loaded IPL Match Data (2008-2024)")
    st.write(df)

    # Chunk the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    # Initialize vectorstore using caching
    vectorstore = initialize_vectorstore(chunks)
    if not vectorstore:
        st.error("Vectorstore could not be initialized.")
        return

    # Set up the retriever
    retriever = vectorstore.as_retriever()

    # Initialize the LLM (Google Generative AI)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.5,
        max_tokens=500,
        timeout=None,
        max_retries=2,
    )

    # Set up the prompt
    system_prompt = (
        "You are an expert in IPL cricket, covering matches, statistics, players, and events from the 2008 season to the 2024 season. "
        "Your goal is to provide accurate and detailed answers based on IPL match data from this period, combining retrieval from a knowledge base with generation of natural, concise responses. "
        "Do all calculations required to answer the question and make sure to generate a SQL query that can be used on the IPL dataset to perform the calculation. "
        "Use the generated SQL query to retrieve the relevant data from the dataset and present it as a dataframe along with the answer. "
        "Prioritize factual accuracy. "
        "When asked for specific match results, provide team names, scores, venues, key player performances, and also retrieve a dataframe of the match details using SQL. "
        "When asked about player statistics, include runs, wickets, averages, and notable performances, and retrieve a dataframe of the player's statistics using SQL. "
        "Always refer to the specific match or season when answering, providing context as needed. "
        "Dont display the sql query, and dont show the python code."
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Create the question-answer chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Streamlit Input Box
    st.subheader("Ask a question about IPL (2008-2024)")
    user_input = st.text_input("Enter your question here:", "")

    # Display the response when the user submits a question
    if user_input:
        response = rag_chain.invoke({"input": user_input})
        st.write("Answer:")
        st.write(response["answer"])


if __name__ == "__main__":
    app()
