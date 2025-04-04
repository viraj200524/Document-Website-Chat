# rag_backend.py - Backend class for the RAG application

# LangChain imports
from langchain_community.document_loaders import WebBaseLoader, TextLoader, PyPDFLoader
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
class RAGBackend:
    def __init__(self, groq_api_key, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the RAG backend with necessary components"""
        self.groq_api_key = groq_api_key
        self.model_name = model_name
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        
        # Initialize text splitter with default values
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
        
        # Initialize document storage
        self.documents = []
        self.vectors = None
        
        # Initialize LLM
        self.llm = self._get_llm()
        
        # Create prompt template
        self.prompt_template ="""
            Answer the Question based on the given context.
            Please Provide accurate answer to the Question based only on the context provided.
            Be concise but thorough in your response.

            <context>
            {context}
            </context>

            Question : {question}
            """

        self.prompt=PromptTemplate(template=self.prompt_template,input_variables=["context","question"])
        self.retriever = None
        self.retriever_chain=None

    def _get_llm(self):
        """Get the Groq LLM instance"""
        return ChatGroq(groq_api_key=self.groq_api_key, model="qwen-2.5-32b")

    def process_file(self, file_path, file_name):
        """Process a file and add it to the knowledge base"""
        try:
            # Process the file based on its extension
            if file_name.lower().endswith('.txt'):
                loader = TextLoader(file_path)
                documents = loader.load()
            elif file_name.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents = loader.load()
            else:
                raise ValueError(f"Unsupported file type: {file_name}")
            
            # Update metadata to include user-friendly source name
            for doc in documents:
                doc.metadata['source'] = file_name
            
            # Add to the document collection
            self.documents.extend(documents)
            
            # Update vector store
            self._update_vector_store(documents)
            
            return True
            
        except Exception as e:
            print(f"Error processing file {file_name}: {str(e)}")
            return False

    def process_url(self, url):
        """Process a URL and add its content to the knowledge base"""
        try:
            # Load the web content
            loader = WebBaseLoader(url)
            documents = loader.load()
            
            # Update metadata for better source tracking
            for doc in documents:
                doc.metadata['source'] = f"URL: {url}"
            
            # Add to document collection
            self.documents.extend(documents)
            
            # Update vector store
            self._update_vector_store(documents)
            
            return True
            
        except Exception as e:
            print(f"Error processing URL {url}: {str(e)}")
            return False

    def _update_vector_store(self, new_documents):
         # Split documents
        splitted_docs = self.text_splitter.split_documents(new_documents)
        
        # Update or create vector store
        if self.vectors is None:
            self.vectors = FAISS.from_documents(splitted_docs, self.embeddings)
        else:
            self.vectors.add_documents(splitted_docs)
        self.retriever = self.vectors.as_retriever(search_kwargs={"k": 5})

        self.retriever_chain=RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt":self.prompt}
        )
        
        return splitted_docs


    def query(self, question, return_sources=False):
        """Query the knowledge base with a question"""
        if not self.vectors:
            raise ValueError("No documents in knowledge base. Please add content first.")
        
        # Get response
        response_obj = self.retriever_chain.invoke({'query': question})
        
        # Extract relevant docs for source tracking
        docs = response_obj.get('source_documents', [])

        # Return format depends on whether source tracking is needed
        if return_sources:
            return response_obj['result'], docs
        else:
            return response_obj['result']

    def clear_knowledge_base(self):
        """Clear the knowledge base"""
        self.vectors = None
        self.documents = []
        return True

    def get_document_count(self):
        """Get the count of documents in the knowledge base"""
        return len(self.documents)