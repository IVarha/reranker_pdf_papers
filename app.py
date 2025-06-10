import gradio as gr
import torch
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from retrieval.rerank import search_and_rerank

# Initialize the model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.3"  # Open source model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512*10,
    temperature=0.4,
    top_p=0.95,
    repetition_penalty=1.15
)

# Create LangChain wrapper
llm = HuggingFacePipeline(pipeline=pipe)

def format_references(reranked_docs):
    """Format documents into a numbered reference list."""
    refs = []
    for i, doc in enumerate(reranked_docs, 1):
        source = doc['metadata'].get('source', 'Unknown source')
        refs.append(f"[{i}] {source}")
    return "\n".join(refs)

def answer_question(query):
    reranked = search_and_rerank(query, 5)
    
    # Format context with document numbers
    context_parts = []
    for i, doc in enumerate(reranked, 1):
        context_parts.append(f"Document [{i}]:\n{doc['content']}")
    
    context = "\n\n".join(context_parts)
    references = format_references(reranked)

    prompt = f"""<s>[INST] Use the context below to answer the question. For each piece of information you use, cite the source using [number] format, where number corresponds to the document number.

    Context:
    {context}

    Question: 
    {query}

    Please structure your response with clear citations. [/INST]</s>"""

    res = llm.invoke(prompt)
    if "[/INST]</s>" in res:
        res = res.split("[/INST]</s>")[-1].strip()
    
    # Format the final response with references
    final_response = f"{res}\n\nReferences:\n{references}"
    return final_response

# Gradio interface
demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=3, placeholder="Ask a question about your documents..."),
    outputs=gr.Textbox(lines=10, label="Response with References"),
    title="📚 Mistral Document QA with References",
    description="Ask questions based on your embedded documents. Responses include citations and references.",
    theme="default"
)

if __name__ == "__main__":
    demo.launch()

# print response in markdown format
