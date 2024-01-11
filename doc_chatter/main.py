import os

from llama_index.llms import Gemini
from llama_index.embeddings import GeminiEmbedding
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext

from dotenv import load_dotenv
load_dotenv()


def start_doc_chatter():
    base_llm = Gemini(api_key=os.environ["GEMINI_API_KEY"])
    embed_model = GeminiEmbedding(model_name='models/embedding-001', api_key=os.environ["GEMINI_API_KEY"])
    service_context = ServiceContext.from_defaults(llm=base_llm, embed_model=embed_model)

    documents = SimpleDirectoryReader("data").load_data(show_progress=True)
    index = VectorStoreIndex.from_documents(
        documents, service_context=service_context, show_progress=True
    )

    query_engine = index.as_query_engine()
    while True:
        query = input("Enter query: ")
        response = query_engine.query(query)
        print(response)