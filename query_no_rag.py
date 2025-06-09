from langchain_huggingface import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

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
    max_length=512*8,
    temperature=0.4,
    top_p=0.95,
    repetition_penalty=1.15
)

# Create LangChain wrapper
llm = HuggingFacePipeline(pipeline=pipe)

query = "Think of it step by step. What is DBS?"



# Top-1 or top-3 context

prompt = f"""<s>[INST] Answer the question:

Question:
{query} [/INST]</s>"""

response = llm.invoke(prompt)

# print response in markdown format
print(response)