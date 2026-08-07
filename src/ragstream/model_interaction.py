import re
import tqdm
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

class BaseReasoning:
    """
    Base class for reasoning models, or rather how to interact with them. This class is designed to be extended by specific reasoning
    model implementations.
    """
    def __init__(self, temperature: float = 0.5):
        self.temperature = temperature
        self.llm = ChatOpenAI(
            base_url="http://127.0.0.1:8080/v1",
            api_key="not-needed",
            temperature=temperature,
        )

class BooleanReasoning(BaseReasoning):
    """
    Assumes use of a Reasoning model. One might consider using an instruction-following model instead if you want to 
    be 'pydantic' about the output. As such, we look for the answer in <answer> tags, and the only acceptable values are 'yes' or 'no'.
    """
    def __init__(self, temperature: float = 0.0, additional_instructions: str = ""):
        super().__init__(temperature=temperature)
        system_prompt = """
        You are a strict boolean assistant. You must answer with exactly the word 'Yes' or the word 'No'. 
        First, think through the problem. Then, you MUST output your final answer wrapped in <answer> tags.
        The only acceptable values inside the tags are 'yes' or 'no'. Example: <answer>yes</answer>.
        """
        self.system_prompt = SystemMessage(content=system_prompt + additional_instructions)

    def invoke(self, message: str):
        response = self.llm.invoke([self.system_prompt, HumanMessage(content=message)])
        raw_text = response.content
        match = re.search(r'<answer>\s*(yes|no)\s*</answer>', raw_text, re.IGNORECASE)
        
        if match:
            extracted_answer = match.group(1).lower()
        else:
            extracted_answer = None
        return raw_text, extracted_answer

class Summarisation(BaseReasoning):
    """
    Builds on top of the BaseReasoning class to provide a summarisation interface.
    """
    def __init__(self, temperature: float = 0.7, additional_instructions: str = ""):
        super().__init__(temperature=temperature)
        system_prompt = """
        You are a summarisation assistant. You must provide a concise summary of the provided text.
        """
        self.system_prompt = SystemMessage(content=system_prompt + additional_instructions)

    def invoke(self, message: str):
        response = self.llm.invoke([self.system_prompt, HumanMessage(content=message)])
        return response.content

if __name__ == "__main__":
    
    model = BooleanReasoning()
    questions = ["Respond 'no'.", "Respond 'yes'.", "Is the sky blue?", "Is fire cold?"]
    expected_answers = ["no", "yes", "yes", "no"]

    for question, expected in tqdm.tqdm(zip(questions, expected_answers), total=len(questions)):
        raw_text, answer = model.invoke(HumanMessage(content=question))
        if answer is None:
            tqdm.tqdm.write(f"Question: {question} | Error: Failed to extract <answer> tags.\nRaw output: {raw_text}")
            continue
        tqdm.tqdm.write(f"Question: {question} | Expected: {expected} | Got: {answer}")