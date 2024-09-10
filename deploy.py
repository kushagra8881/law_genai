import streamlit as st
import chromadb
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.load import dumps, loads
import openai
import gdown

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
        st.error(f"Error during vector search: {e}")
        return []

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
        st.error(f"Error during query generation: {e}")
        return []

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
        st.error(f"Error during RRF: {e}")
        return []

# Streamlit UI
st.title("Indian Law Assistant")
st.info("This is a law assistant that can answer questions about Indian law. It uses a vector search to find the most relevant documents and a language model to answer the questions.")
st.divider()
st.image("image.png")
query = st.text_input("Enter your query about Indian law:")
query_length = st.number_input("Number of reformulated queries:", min_value=1, max_value=10, value=3)
api_key = st.text_input("Enter your OpenAI API key:", type="password")

if st.button("Submit"):
    os.environ["OPENAI_API_KEY"] = api_key  # Set the OpenAI API key from user input
    openai.api_key = api_key  # Set the OpenAI API key from user input

    generated_queries = generate_query(query, query_length)

    all_results = []
    for g_query in generated_queries:
        documents = vector_search(g_query, top_k=5)
        all_results.append(documents)
    
    fused_results = reciprocal_rank_fusion(all_results)

    # Display results
    
    
    # Convert results to string format
    fused_results_str = "\n".join([f"Document: {result}, Score: {score}" for result, score in fused_results])

    # Define the prompt template
    prompt = PromptTemplate(
        input_variables=["query", "fused_results"],
        template="""
        You are a helpful assistant that can answer questions about Indian law.
        You are given a query: "{query}" and the following fused results from a vector search:

        {fused_results}
        These are the results from the vector search. Take the best result and provide a response.
        """
    )

    # Format the prompt
    formatted_prompt = prompt.format(
        query=query,
        fused_results=fused_results_str
    )

    # Get the OpenAI response
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": formatted_prompt}
        ],
        max_tokens=300,
        temperature=0.7
    )

    # Print the response
    st.write(response.choices[0].message['content'])

    st.divider()
    st.title("Fused Results")

    for i in fused_results:
        st.write(i)