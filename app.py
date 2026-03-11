import os
from dotenv import load_dotenv
from rag_pipeline import build_vector_db
from utils import retrieve_context, build_prompt
from langchain_openai import ChatOpenAI

load_dotenv()

def main():
    papers_folder = "papers"
    db = build_vector_db(papers_folder)
    
    if db is None:
        return

    # Low temperature = higher accuracy for technical papers
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

    print("\n--- Research Assistant Ready ---")
    print("Analyizing papers in '/papers' folder...")

    while True:
        question = input("\nQuestion (or 'exit'): ")
        if question.lower() == "exit":
            break

        context, results = retrieve_context(db, question)
        prompt = build_prompt(context, question)
        
        # Displaying which pages the snippets came from to help you verify
        print(f"\n[Searching {len(results)} relevant snippets...]")
        
        response = llm.invoke(prompt)

        print("\n" + "="*60)
        print(response.content)
        print("="*60)

if __name__ == "__main__":
    main()