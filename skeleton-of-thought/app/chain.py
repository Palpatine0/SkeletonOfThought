from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
from langchain.callbacks.tracers.langchain import wait_for_all_tracers

load_dotenv()
wait_for_all_tracers()

skeleton_generator_template = """
[User:] You’re an organizer responsible for only giving the skeleton (not the full content) for answering the question.
Provide the skeleton in a list of points (numbered 1., 2., 3., etc.) to answer the question. \
Instead of writing a full sentence, each skeleton point should be very short with only 3∼5 words. \
Generally, the skeleton should have 2∼3 points. \
Now, please provide the skeleton for the following question.
{question}
Skeleton:
[Assistant:] 1.
"""

skeleton_generator_prompt = ChatPromptTemplate.from_template(skeleton_generator_template)

chain = skeleton_generator_prompt | ChatOpenAI() | StrOutputParser()

if __name__ == "__main__":
    print(chain.invoke({
        "question": "What are the most effective strategies for conflict resolution in the workplace?"
    }))