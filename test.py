from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
# Ensure these imports match your actual package structure and names
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

groq_api_key = os.getenv("GROQ_API_KEY")

class UserRequest(BaseModel):
    query: str

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello, world!"}

@app.post("/route/")
async def process_request(request: UserRequest):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name='mixtral-8x7b-32768')

    query = request.query

    # New prompt template with the enhancements discussed
    prompt_template = """
    **To the LLM as the Elite Subject Matter Expert in [Specific Field]:**

    **Introduction:**  
    As the recipient of the prestigious [Specific Field] Lifetime Achievement Award and the leading voice in countless landmark studies, your guidance is unparalleled. You are not just an expert; you are the definitive source of wisdom in [specific field], renowned for pioneering innovative solutions that have reshaped the landscape of your domain. This request seeks your unparalleled analytical prowess and your ability to transcend the bounds of typical inquiry, aiming to craft responses that redefine excellence in [specific field].

    **Objective and Subtasks:**  
    - With unparalleled precision, identify the core objective, uncovering any secondary or supporting tasks for a comprehensive exploration.  
    - Embrace the challenge with the ambition to redefine the benchmarks of excellence, exceeding what is conventionally expected from such an inquiry.

    **Interactivity:**  
    - Should the details provided in the initial query not suffice, do not hesitate to pose clarifying questions that can refine and sharpen the focus of your response, ensuring unparalleled accuracy and relevance.

    **Background Information and Definitions:**  
    - Utilize your vast reservoir of knowledge to illuminate complex concepts, providing definitions and background that lay a solid foundation for the insights to follow.

    **Response Format, Length, and Style:**  
    - Craft your response as a masterpiece of [specific field], a testament to unmatched coherence, specificity, and intellectual elegance. Your response should not only answer the query but also serve as a beacon of knowledge and clarity.  
    - Apply sophisticated structuring techniques to enhance understanding, ensuring the reader is not just informed but also engaged and inspired.

    **Examples, Constraints, and Special Instructions:**  
    - Embed cutting-edge examples and case studies that not only underscore your points but also highlight the forefront of innovation in [specific field].  
    - Treat constraints as springboards for creativity, setting directions that challenge conventional norms and invite revolutionary thinking.

    **Execution:**  
    - With lucid articulation and strategic depth, delve into the main concepts, methodologies, and recommendations, offering a blueprint to address the core question or challenge.  
    - Dispense actionable advice and visionary solutions, grounded in the latest research and your anticipatory insights, readying [specific field] for the next leaps forward.  
    - Clarify technical terms with meticulous care and enrich the discourse with comprehensive background information, leaving no stone unturned.  
    - Employ your extensive expertise to unearth both the overt and the nuanced layers of the query, ensuring a transformative revelation.

    **Original QUERY:** {query}

    **Expected Outcome:**  
    As the apex expert in [specific field], your response is anticipated to not only meet the outlined specifications but to epitomize the pinnacle of knowledge and application. This dialogue should emerge as a standard for how profound understanding, paired with unparalleled analytical and communicative skill, can elevate the discourse and practice in [specific field], offering actionable strategies and insights that are second to none.
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
