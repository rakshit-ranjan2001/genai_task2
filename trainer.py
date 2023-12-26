import openai
import pandas as pd
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders.dataframe import DataFrameLoader
from langchain.vectorstores import Pinecone
import dotenv
import os

_ = dotenv.load_dotenv()
pinecone.init(api_key=os.getenv("Pinecone_key"), environment="gcp-starter")
index = pinecone.Index("practice-index")
delete_pinecone = index.delete(delete_all=True)
df = pd.read_csv("gardening_dataset.csv")
loader = DataFrameLoader(df["text"].to_frame(name="text"))
documents = loader.load()
embeddings = OpenAIEmbeddings()
vectorstore = Pinecone.from_existing_index(
    index_name="practice-index", embedding=embeddings, text_key="text"
)
vectorstore.add_documents(documents)
