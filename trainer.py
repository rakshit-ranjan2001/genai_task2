import openai
import pandas as pd
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import dotenv
import os

_ = dotenv.load_dotenv()
pinecone.init(api_key=os.getenv("Pinecone_key"), environment="gcp-starter")
df = pd.read_csv("gardening_dataset.csv")
index = pinecone.Index("practice-index")
embeddings = OpenAIEmbeddings()

upserted_data = []
for i, item in enumerate(df["text"].to_list()):
    upserted_data.append((str(i), embeddings.embed_query(item), {"text": item}))
    print(f"Created vector {i}")
    if i % 100 == 0:
        index.upsert(vectors=upserted_data)
        upserted_data = []
        print(f"Upserted vectors {i-100} to {i}")
index.upsert(vectors=upserted_data)
print("Upserted remaining vectors")
