import openai
import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import dotenv
import os

_ = dotenv.load_dotenv()
embeddings = OpenAIEmbeddings()
pinecone.init(api_key=os.getenv("Pinecone_key"), environment="gcp-starter")
docsearch = Pinecone.from_existing_index(
    index_name="practice-index", embedding=embeddings
)
chat = ChatOpenAI()


def get_reply(prompt: str) -> str:
    pinecone_replies = docsearch.similarity_search(query=prompt, k=3)
    context = ""
    for i in pinecone_replies:
        context += i.page_content
    template = "You are a gardening assistant bot, an automated service that answers \
    the queries of the people interested in gardening. Greet the user first in a \
    friendly manner and then answer the question as truthfully as possible using the \
    provided context, and if the answer is not contained within the context and requires \
    some latest information to be updated, print 'Sorry Not Sufficient context to answer query' The context is as follows:\
    {context}"
    prompt_template = PromptTemplate.from_template(template)
    prompt_message = prompt_template.format(context=context)
    print(prompt_message)
    messages.append(SystemMessage(content=prompt_message))
    messages.append(HumanMessage(content=prompt))
    reply = chat(messages)
    return reply.content


if __name__ == "__main__":
    messages = []
    while True:
        prompt = input("User: ")
        if prompt == "exit":
            print("Thank you for gardening with us.")
            break
        res = get_reply(prompt)
        print(f"Assistant: {res}\n\n")
