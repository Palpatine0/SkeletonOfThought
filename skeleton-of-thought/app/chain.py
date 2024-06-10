from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
from langchain.callbacks.tracers.langchain import wait_for_all_tracers

load_dotenv()
wait_for_all_tracers()

# skeleton_generator
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

skeleton_generator_chain = skeleton_generator_prompt | ChatOpenAI() | StrOutputParser()

# point_expander
point_expander_template = """
[User:] You’re responsible for continuing the writing of one and only one point in the overall answer to the following question.
{question}
The skeleton of the answer is
{skeleton}
Continue and only continue the writing of point {point_index}. 
Write it **very shortly** in 1∼2 sentence and do not continue with other points!
[Assistant:] {point_index}. {point_skeleton}
"""

point_expander_prompt = ChatPromptTemplate.from_template(point_expander_template)

point_expander_chain = point_expander_prompt | ChatOpenAI() | StrOutputParser()


def parse_numbered_list(input_str):
    """
    Parses a numbered list into a list of dictionaries with each element having two keys:
    'index' for the index in the numbered list, and 'point' for the content.
    """
    lines = input_str.split('\n')

    parsed_list = []

    for line in lines:
        parts = line.split('. ', 1)

        if len(parts) == 2:
            index = int(parts[0])
            point = parts[1].strip()
            parsed_list.append({'point_index': index, 'point_skeleton': point})

    return parsed_list


def create_list_elements(_input):
    skeleton = _input['skeleton']
    numbered_list = parse_numbered_list(skeleton)
    for el in numbered_list:
        el["skeleton"] = skeleton
        el["question"] = _input['question']
    return numbered_list


chain = RunnablePassthrough().assign(
    skeleton = skeleton_generator_chain
) | create_list_elements | point_expander_chain.map() | (lambda x: "\n\n".join(x))

if __name__ == "__main__":
    print(chain.invoke({
        "question": "What are the most effective strategies for conflict resolution in the workplace?",
    }))
