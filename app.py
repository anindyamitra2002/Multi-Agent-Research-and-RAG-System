import streamlit as st
import asyncio
from src.vectorstore.pinecone_db import ingest_data, get_retriever, load_documents, process_chunks, save_to_parquet
from src.agents.research_agent import create_industry_research_workflow
from src.agents.workflow import run_adaptive_rag
from pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.pregel import GraphRecursionError
import tempfile
import os
import time
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Research & RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
    }
    .config-section {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .chat-container {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "pinecone_client" not in st.session_state:
    st.session_state.pinecone_client = None
if "research_config_saved" not in st.session_state:
    st.session_state.research_config_saved = False
if "rag_config_saved" not in st.session_state:
    st.session_state.rag_config_saved = False

def save_research_config(api_keys):
    """Save research configuration."""
    st.session_state.research_openai_key = api_keys['openai']
    st.session_state.research_tavily_key = api_keys['tavily']
    st.session_state.research_config_saved = True


def research_config_section():
    """Configuration section for Company Research tab."""
    st.markdown("### ⚙️ Configuration")
    
    with st.expander("API Configuration", expanded=not st.session_state.research_config_saved):
        col1, col2 = st.columns(2)
        with col1:
            openai_key = st.text_input(
                "OpenAI API Key",
                type="password",
                value=st.session_state.get('research_openai_key', ''),
                key="research_openai_input"
            )
        with col2:
            tavily_key = st.text_input(
                "Tavily API Key",
                type="password",
                value=st.session_state.get('research_tavily_key', ''),
                key="research_tavily_input"
            )
        
        if st.button("Save Research Configuration", key="save_research_config"):
            if openai_key and tavily_key:
                save_research_config({
                    'openai': openai_key,
                    'tavily': tavily_key
                })
                if not os.environ.get("TAVILY_API_KEY"):
                    os.environ["TAVILY_API_KEY"] = tavily_key
                st.success("✅ Research configuration saved!")
            else:
                st.error("Please provide both API keys.")


async def run_industry_research(company: str, industry: str, llm):
    """Run the industry research workflow asynchronously."""
    workflow = create_industry_research_workflow(llm)
    
    output = await workflow.ainvoke(input={
        "company": company,
        "industry": industry
    }, config={"recursion_limit": 5})
    
    return output['final_report']


def research_input_section():
    """Input section for Company Research tab."""
    st.markdown("### 🔍 Research Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        company_name = st.text_input(
            "Company Name",
            placeholder="e.g., Tesla",
            help="Enter the name of the company to research"
        )
    with col2:
        industry_type = st.text_input(
            "Industry Type",
            placeholder="e.g., Automotive",
            help="Enter the industry sector"
        )
    
    if st.button("Generate Research Report", 
                 disabled=not st.session_state.research_config_saved,
                 type="primary"):
        if company_name and industry_type:
            with st.spinner("🔄 Generating comprehensive research report..."):
                # try:
                # Initialize LLM and run research
                llm = ChatOpenAI(
                    model="gpt-3.5-turbo-0125",
                    temperature=0.1,
                    api_key=st.session_state.research_openai_key
                )
                
                report_path = asyncio.run(run_industry_research(
                    company=company_name,
                    industry=industry_type,
                    llm=llm
                ))
                
                if os.path.exists(report_path):
                    with open(report_path, "rb") as file:
                        st.download_button(
                            "📥 Download Research Report",
                            data=file,
                            file_name=f"{company_name}_research_report.pdf",
                            mime="application/pdf"
                        )
                else:
                    st.error("Report generation failed.")
                # except Exception as e:
                #     st.error(f"Error during report generation: {str(e)}")
        else:
            st.warning("Please fill in both company name and industry type.")
            
def initialize_pinecone(api_key):
    """Initialize Pinecone client with API key."""
    try:
        return Pinecone(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing Pinecone: {str(e)}")
        return None

def initialize_llm(llm_option, openai_api_key=None):
    """Initialize LLM based on user selection."""
    if llm_option == "OpenAI":
        if not openai_api_key:
            st.sidebar.warning("Please enter OpenAI API key.")
            return None
        return ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

def clear_pinecone_index(pc, index_name="vector-index"):
    """Clear the Pinecone index."""
    try:
        pc.delete_index(index_name)
        st.session_state.documents_processed = False
        st.session_state.retriever = None
        st.success("Database cleared successfully!")
    except Exception as e:
        st.error(f"Error clearing database: {str(e)}")

def process_documents(uploaded_files, pc):
    """Process uploaded documents and store in Pinecone."""
    if not uploaded_files:
        st.warning("Please upload at least one document.")
        return False

    with st.spinner("Processing documents..."):
        temp_dir = tempfile.mkdtemp()
        file_paths = []
        markdown_path = Path(temp_dir) / "combined.md"
        parquet_path = Path(temp_dir) / "documents.parquet"
        
        for uploaded_file in uploaded_files:
            file_path = Path(temp_dir) / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            file_paths.append(str(file_path))

        try:
            markdown_path = load_documents(file_paths, output_path=markdown_path)
            chunks = process_chunks(markdown_path, chunk_size=256, threshold=0.6)
            print(f"Processed chunks: {chunks}")
            parquet_path = save_to_parquet(chunks, parquet_path)
            
            ingest_data(
                pc=pc,
                parquet_path=parquet_path,
                text_column="text",
                pinecone_client=pc
            )
            
            st.session_state.retriever = get_retriever(pc)
            st.session_state.documents_processed = True
            
            return True
            
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            return False
        finally:
            for file_path in file_paths:
                try:
                    os.remove(file_path)
                except:
                    pass
            try:
                os.rmdir(temp_dir)
            except:
                pass

def run_rag_with_streaming(retriever, question, llm, enable_web_search=False):
    """Run RAG workflow and yield streaming results."""
    try:
        response = run_adaptive_rag(
            retriever=retriever,
            question=question,
            llm=llm,
            top_k=5,
            enable_websearch=enable_web_search
        )
        
        for word in response.split():
            yield word + " "
            time.sleep(0.03)
            
    except GraphRecursionError:
        response = "I apologize, but I cannot find a sufficient answer to your question in the provided documents. Please try rephrasing your question or ask something else about the content of the documents."
        for word in response.split():
            yield word + " "
            time.sleep(0.03)
            
    except Exception as e:
        yield f"I encountered an error while processing your question: {str(e)}"


def document_upload_section():
    """Document upload section for RAG tab."""
    st.markdown("### 📄 Document Management")
    
    if not st.session_state.documents_processed:
        uploaded_files = st.file_uploader(
            "Upload your documents",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt", "pptx", "md"],
            help="Support multiple file uploads"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            if uploaded_files:
                st.info(f"📁 {len(uploaded_files)} files selected")
        with col2:
            if st.button(
                "Process Documents",
                disabled=not (uploaded_files and st.session_state.rag_config_saved)
            ):
                if process_documents(uploaded_files, st.session_state.pinecone_client):
                    st.success("✅ Documents processed successfully!")
    else:
        st.success("✅ Documents are loaded and ready for querying!")
        if st.button("Upload New Documents"):
            st.session_state.documents_processed = False
            st.rerun()

# Update the save_rag_config function to remove web_search
def save_rag_config(config):
    """Save RAG configuration."""
    st.session_state.rag_pinecone_key = config['pinecone']
    st.session_state.rag_openai_key = config['openai']
    st.session_state.rag_config_saved = True

# Update the rag_config_section to remove web_search checkbox
def rag_config_section():
    """Configuration section for RAG tab."""
    st.markdown("### ⚙️ Configuration")
    
    with st.expander("API Configuration", expanded=not st.session_state.rag_config_saved):
        col1, col2 = st.columns(2)
        with col1:
            pinecone_key = st.text_input(
                "Pinecone API Key",
                type="password",
                value=st.session_state.get('rag_pinecone_key', ''),
                key="rag_pinecone_input"
            )
        with col2:
            openai_key = st.text_input(
                "OpenAI API Key",
                type="password",
                value=st.session_state.get('rag_openai_key', ''),
                key="rag_openai_input"
            )
        
        if st.button("Save RAG Configuration", key="save_rag_config"):
            if pinecone_key and openai_key:
                save_rag_config({
                    'pinecone': pinecone_key,
                    'openai': openai_key
                })
                # Initialize Pinecone client
                st.session_state.pinecone_client = initialize_pinecone(pinecone_key)
                st.success("✅ RAG configuration saved!")
            else:
                st.error("Please provide both API keys.")

# Update the chat_interface function to include web search toggle
def chat_interface():
    """Enhanced chat interface with streaming responses and web search toggle."""
    st.markdown("### 💬 Chat Interface")
    
    # Add web search toggle in the chat interface
    col1, col2 = st.columns([3, 1])
    with col2:
        web_search = st.checkbox(
            "🌐 Enable Web Search",
            value=st.session_state.get('use_web_search', False),
            help="Toggle web search for additional context",
            key="web_search_toggle"
        )
    st.session_state.use_web_search = web_search
    
    # Chat container with messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input(
        "Ask a question about your documents...",
        disabled=not st.session_state.documents_processed,
        key="chat_input"
    ):
        # User message
        with st.chat_message("user"):
            if st.session_state.use_web_search:
                st.markdown(f"{prompt} 🌐")
            else:
                st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Assistant response
        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""
            
            try:
                with st.spinner("Thinking..."):
                    llm = ChatOpenAI(
                        api_key=st.session_state.rag_openai_key,
                        model="gpt-3.5-turbo"
                    )
                    
                    for chunk in run_rag_with_streaming(
                        retriever=st.session_state.retriever,
                        question=prompt,
                        llm=llm,
                        enable_web_search=st.session_state.use_web_search
                    ):
                        full_response += chunk
                        response_container.markdown(full_response + "▌")
                
                response_container.markdown(full_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
            
            except Exception as e:
                st.error(f"Error: {str(e)}")

def main():
    """Main application layout."""
    st.title("🤖 Research & RAG Assistant")
    
    tab1, tab2 = st.tabs(["🔍 Company Research", "💬 Document Q&A"])
    
    with tab1:
        research_config_section()
        if st.session_state.research_config_saved:
            st.divider()
            research_input_section()
        else:
            st.info("👆 Please configure your API keys above to get started.")
    
    with tab2:
        rag_config_section()
        if st.session_state.rag_config_saved:
            st.divider()
            document_upload_section()
            if st.session_state.documents_processed:
                st.divider()
                chat_interface()
        else:
            st.info("👆 Please configure your API keys above to get started.")

if __name__ == "__main__":
    main()