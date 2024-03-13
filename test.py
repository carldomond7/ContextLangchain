from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
# Make sure these imports match your actual package structure and names
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

groq_api_key = os.getenv("GROQ_API_KEY")

class UserRequest(BaseModel):
    query: str

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "plswork!"}

@app.post("/route/")
async def process_request(request: UserRequest):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name='mixtral-8x7b-32768')

    query = request.query

    # Updated prompt template with ultra-refined enhanced prompt
    prompt_template = """
    **To the LLM as the Elite Subject Matter Expert in [Specific Field]:**

    **Introduction:**
    As the unrivaled authority and the most sought-after source of knowledge in [specific field], you are presented with a request that demands not just your expertise, but your unique capability to transcend typical response boundaries. This task requires an enhanced prompt that meticulously addresses key elements, infused with additional context and directives to magnify the output quality of a large language model (LLM). Your unparalleled insight into [specific field] will be pivotal in crafting a response that sets new standards for clarity, depth, and applicability.

    **Objective and Subtasks:**
    - Pinpoint the core objective with precision, identifying any underlying or supplementary tasks to provide a holistic analysis.
    - Approach these tasks with the intent to not only meet but to exceed the conventional expectations for such inquiries.

    **Background Information and Definitions:**
    - Draw upon your exhaustive repository of knowledge to supply essential background information, offering definitions that illuminate complex concepts with ease.

    **Response Format, Length, and Style:**
    - Dictate the format, articulating a response that embodies the pinnacle of coherence, specificity, and intellectual sophistication. Aim for a response that, through its structure and substance, becomes a paragon in [specific field].
    - Encourage the utilization of advanced structuring techniques, such as logical segmentation, annotated guides, or methodical breakdowns, to elevate comprehension and reader engagement.

    **Examples, Constraints, and Special Instructions:**
    - Integrate exemplars that not only illustrate your points but also showcase cutting-edge applications and theoretical breakthroughs.
    - Highlight constraints as opportunities for innovative thinking and specify instructions that challenge the norm, promoting a narrative that reshapes understanding and application in [specific field].

    **Execution:**
    - With unmatched clarity and an expertly curated approach, elucidate the main concepts, methodologies, or recommendations that answer the core question or fulfill the primary objective.
    - Provide actionable advice and forward-thinking solutions, leveraging the forefront of research, developments, and your predictive insights into future trends or challenges.
    - Precisely define technical terms and offer expansive background information, ensuring no nuance is overlooked.
    - Utilize your comprehensive expertise to address not just the explicit but also the implicit facets of the query, delivering a response that is as enlightening as it is transformative.

    **Original QUERY:** {query}

    **Expected Outcome:**
    As the definitive expert in [specific field], your response should not only adhere to the specifications outlined above but also personify the zenith of knowledge and application in the domain. This answer should serve as a benchmark, demonstrating how profound understanding, coupled with an elite level of analytical and communicative prowess, can transcend typical response paradigms to offer unparalleled insights and actionable strategies.
    """

    # Define the prompt structure
    prompt = PromptTemplate(
        input_variables=["query"],
        template=prompt_template,
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Pass the context and question to the Langchain chain
    result_chain = llm_chain.invoke({"query": query})

    return result_chain

if __name__ == "__main__":
    uvicorn.run(app)
