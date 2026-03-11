def retrieve_context(db, question, k=5): # Increased k for better coverage
    results = db.similarity_search(question, k=k)
    context = ""
    for r in results:
        context += r.page_content + "\n"
    return context, results

def build_prompt(context, question):
    # Expert-level persona helps with IC design and sensor topics
    prompt = f"""
You are an expert Research Assistant specializing in Electrical Engineering and Integrated Circuits.

Use the following context from research papers to provide a detailed, technical answer to the question. 
If the information is not present in the context, state that you do not have enough information rather than guessing.

Context:
{context}

Question:
{question}

Technical Answer:
"""
    return prompt