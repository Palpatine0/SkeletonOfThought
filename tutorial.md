## Skeleton of Thought

### Project Introduction
This project aims to significantly speed up the response generation of LLMs using parallel decoding techniques. Inspired by the "Skeleton of Thought" paper, it introduces a method to generate concise, structured responses that are expanded into detailed answers. Implemented within the LangChain framework, this approach breaks down complex queries into smaller chunks processed simultaneously, drastically reducing processing time and enhancing overall LLM performance. This project exemplifies the potential of advanced LLM techniques to make information processing faster and more efficient.
 
### Prerequisites
- Python 3.11
- pip (Python package installer)
- Git (optional)

### Step 1: Initial Setup

#### 1. Initialize the Environment
First, let's set up the environment and install necessary dependencies.


1. **Create a `.env` file:**
   - This file will store your API keys and other configuration settings. Ensure it is included in your `.gitignore` file to prevent it from being committed to your repository.

   Example `.env` file:
   ```plaintext
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
   LANGCHAIN_API_KEY="your_langchain_api_key"
   LANGCHAIN_PROJECT="SkeletonOfThought"
   OPENAI_API_KEY="your_open_api_key"
   ```
   

2. **Install required packages:**
   ```bash
   pip install langchain langchain_community openai streamlit python-dotenv
   ```
   ```bash
   pip install -U langchain-cli
   ```
 
### Step 2: Setup LangServe and LangSmith

#### 1. LangServe Setup
Set up LangServe to manage our application deployment.

1. **Initialize a New LangServe Application:**
   - Use the LangServe CLI to create a new application called `sql-research-assistant`.

   Command:
   ```bash
   langchain app new sql-research-assistant
   ```
#### 2. LangSmith Setup

Make sure u have created a LangSmith project for this lab.

**Project Name:** SkeletonOfThought


### Step 3: Implement the Skeleton Generator Chain

#### 1. Create the `chain.py` File
Create the `chain.py` file within your project directory to implement the skeleton generator chain for conflict resolution strategies.

**File**: `skeleton-of-thought/app/chain.py`

**Code for `chain.py`:**
```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.callbacks.tracers.langchain import wait_for_all_tracers

load_dotenv()
wait_for_all_tracers()

# skeleton_generator
skeleton_generator_template = """
[User:] You’re an organizer responsible for only giving the skeleton (not the full content) for answering the question.
Provide the skeleton in a list of points (numbered 1., 2., 3., etc.) to answer the question. 
Instead of writing a full sentence, each skeleton point should be very short with only 3∼5 words. 
Generally, the skeleton should have 2∼3 points. 
Now, please provide the skeleton for the following question.
{question}
Skeleton:
[Assistant:] 1.
"""

skeleton_generator_prompt = ChatPromptTemplate.from_template(skeleton_generator_template)

skeleton_generator_chain = skeleton_generator_prompt | ChatOpenAI() | StrOutputParser()

if __name__ == "__main__":
    print(skeleton_generator_chain.invoke({
        "question": "What are the most effective strategies for conflict resolution in the workplace?"
    }))
```

Adjust the template as needed, like `3∼5 words`, or `2∼3 points` in the template 

#### 2. Test the Skeleton Generator Chain
Run the `chain.py` file to ensure that the skeleton generator chain produces the expected output. This step verifies that the chain correctly generates a skeleton of key points in response to a given question.

After running the `chain.py` file, you should see output **similar** to the following:

<img src="https://i.imghippo.com/files/g3YOD1718051975.jpg" alt="" border="0">

This output demonstrates that the skeleton generator chain is functioning correctly, providing concise, structured responses to the input question.


### Step 4: Implement the Point Expander Chain

#### 1. Add the Point Expander Chain
Enhance the `chain.py` file to include a point expander chain, which elaborates on individual points from the skeleton.

**Updated Code for `chain.py`:**
```python
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
Provide the skeleton in a list of points (numbered 1., 2., 3., etc.) to answer the question. 
Instead of writing a full sentence, each skeleton point should be very short with only 3∼5 words. 
Generally, the skeleton should have 2∼3 points. 
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

if __name__ == "__main__":
    skeleton = """
    1. Communication
    2. Active listening
    3. Collaboration
    4. Mediation
    5. Conflict coaching
    6. Establishing ground rules
    7. Seeking common ground
    """
    print(skeleton_generator_chain.invoke({
        "question": "What are the most effective strategies for conflict resolution in the workplace?"
    }))
    print(point_expander_chain.invoke({
        "question": "What are the most effective strategies for conflict resolution in the workplace?",
        "skeleton": skeleton,
        "point_index": "1",
        "point_skeleton": "Communication."
    }))
```

Adjust the template as needed, like `3∼5 words`, or `2∼3 points` in the template

#### 2. Test the Point Expander Chain
Run the updated `chain.py` file to ensure that the point expander chain elaborates on individual points from the skeleton as expected.

After running the `chain.py` file, you should see output similar to the following:

<img src="https://i.imghippo.com/files/wlecT1718053354.jpg" alt="" border="0">

This output demonstrates that the point expander chain is functioning correctly, **providing short, elaborated responses** for each individual point in the skeleton.


### Step 5: Combine Chains for Comprehensive Skeleton Generation

#### 1. Add Utility Functions and Combine Chains
Enhance the `chain.py` file by including utility functions to parse numbered lists generated by skeleton generator and create elements for the point expander. 
Combine the chains to generate a skeleton, parse points, and expand each point using `RunnablePassthrough` and mapping.

**Updated Code for `chain.py`:**
```python
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

def get_final_answer(expanded_list):
    final_answer_str = "Here's a comprehensive answer:\n\n"
    for i, el in enumerate(expanded_list):
        final_answer_str += f"{i+1}. {el}\n\n"
    return final_answer_str

chain = RunnablePassthrough().assign(
    skeleton = skeleton_generator_chain
) | create_list_elements | point_expander_chain.map() | get_final_answer

if __name__ == "__main__":
    print(chain.invoke({
        "question": "What are the most effective strategies for conflict resolution in the workplace?",
    }))
```

<img src="https://i.imghippo.com/files/M2p1q1718055908.jpg" alt="" border="0">

#### 2. Enhance point expander chain to concatenate expanded points
```python
point_expander_chain = RunnablePassthrough.assign(
    continuation = point_expander_prompt | ChatOpenAI() | StrOutputParser()
) | (lambda x: x["point_skeleton"].strip() + " " + x['continuation'])
```
<img src="https://i.imghippo.com/files/JOHzB1718057050.jpg" alt="" border="0">


#### 3. Test the Comprehensive Chain
Run the updated `chain.py` file to ensure that the combined chains work together to generate and expand each point in the skeleton comprehensively.

#### Example Output
After running the `chain.py` file, you should see output similar to the following:

<img src="https://i.imghippo.com/files/G2Pn91718057123.jpg" alt="" border="0">

This output demonstrates that the combined chains are functioning correctly, 
**the model first generated a comprehensive skeleton and then expanding each point** as expected.

### Step 5: Serve the Application Using LangServe

#### 1. Update `server.py`:

<img src="https://i.imghippo.com/files/fT16o1718058478.jpg" alt="" border="0">

#### 2. Update `chain.py`:
<img src="https://i.imghippo.com/files/WCeQR1718058257.jpg" alt="" border="0">

#### 3. Serving the Application by LangServe

Run the following commands to set up and serve the application using LangServe.

   ```bash
   cd skeleton-of-thought
   langchain serve
   ```

Access [Prompter Playground](http://127.0.0.1:8000/skeleton-of-thought/playground/), be note that the port may vary depending on the configuration.

<img src="https://i.imghippo.com/files/BDwbr1718058672.jpg" alt="" border="0">