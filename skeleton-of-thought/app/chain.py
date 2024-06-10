from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant who speaks like a pirate",
        ),
        ("human", "{text}"),
    ]
)

_model = ChatOpenAI()

chain = _prompt | _model
