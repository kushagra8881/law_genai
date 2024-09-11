import gradio as gr
import chromadb
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.load import dumps, loads
import openai

# Initialize the ChromaDB client
client = chromadb.PersistentClient(path="indian_law_bge_work_1")

# Load the collection
collection = client.get_or_create_collection("indian_law_bge_work_1")

# Vector Search Function
def vector_search(query, top_k=5):
    try:
        results = collection.query(query_texts=[query], n_results=top_k)
        return results['documents']
    except Exception as e:
        return f"Error during vector search: {e}"

# Generate Query Function
def generate_query(query, query_length):
    try:
        prompt = PromptTemplate(
            input_variables=["query", "query_length"],
            template="""
            You are a helpful assistant that can answer questions about Indian law.
            You are given a query: "{query}" and you need to generate {query_length} reformulated queries for vector search.
            """
        )

        llm = OpenAI(temperature=0.7)
        chain = LLMChain(llm=llm, prompt=prompt, output_parser=StrOutputParser())

        result = chain.run({"query": query, "query_length": query_length})
        result = [i.strip() for i in result.split("\n") if i.strip()]
        result = [i for i in result if i != ""]
        return result
    except Exception as e:
        return f"Error during query generation: {e}"

# Reciprocal Rank Fusion Function
def reciprocal_rank_fusion(results_list, k=60):
    fused_scores = {}

    try:
        for docs in results_list:
            for rank, doc in enumerate(docs):
                doc_str = dumps(doc)
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                fused_scores[doc_str] += 1 / (rank + 1 + k)

        reranked_results = [
            (loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return reranked_results
    except Exception as e:
        return f"Error during RRF: {e}"

# Main Function to Handle the Workflow
def handle_query(openai_key, query, query_length):
    openai.api_key = openai_key
    os.environ["OPENAI_API_KEY"] = openai_key
    
    # Generate reformulated queries
    generated_queries = generate_query(query, query_length)
    
    if isinstance(generated_queries, str):
        return generated_queries, []
    
    all_results = []
    for g_query in generated_queries:
        documents = vector_search(g_query, top_k=5)
        if isinstance(documents, str):  # Error handling
            return documents, []
        all_results.append(documents)
    
    # Fuse results using RRF
    fused_results = reciprocal_rank_fusion(all_results)
    
    if isinstance(fused_results, str):
        return fused_results, []
    
    # Prepare fused results for language model input
    fused_results_str = "\n".join([f"Document: {result}, Score: {score}" for result, score in fused_results])

    prompt = PromptTemplate(
        input_variables=["query", "fused_results"],
        template="""
        You are a helpful assistant that can answer questions about Indian law.
        You are given a query: "{query}" and the following fused results from a vector search:

        {fused_results}
        These are the results from the vector search. Take the best result and provide a response.
        """
    )

    formatted_prompt = prompt.format(
        query=query,
        fused_results=fused_results_str
    )

    # Get the OpenAI response
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": formatted_prompt}
        ],
        max_tokens=300,
        temperature=0.7
    )

    answer = response.choices[0].message['content']
    
    return answer, fused_results

# Gradio Interface
def app(openai_key, query, query_length):
    answer, fused_results = handle_query(openai_key, query, query_length)
    
    fused_results_str = "\n".join([f"Document: {result}, Score: {score}" for result, score in fused_results])
    
    return answer, fused_results_str

with gr.Blocks() as demo:
    gr.Markdown("## Indian Law Assistant")
    
    openai_key = gr.Textbox(label="OpenAI API Key", placeholder="Enter your OpenAI API key")
    query = gr.Textbox(label="Query", placeholder="Enter your query about Indian law")
    query_length = gr.Slider(minimum=1, maximum=10, value=3, label="Number of Reformulated Queries")

    answer_output = gr.Textbox(label="Answer", interactive=False)
    fused_results_output = gr.Textbox(label="Fused Results", interactive=False)

    submit_button = gr.Button("Submit")
    
    submit_button.click(
        fn=app,
        inputs=[openai_key, query, query_length],
        outputs=[answer_output, fused_results_output]
    )

demo.launch()
