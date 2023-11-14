from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

api_key = "AIzaSyDJo3U8reDx1xwtQPgZtRIrBjBffq_KEdM"
llm = GooglePalm(google_api_key = api_key, temperature=0.7 )

embeddings = HuggingFaceInstructEmbeddings(
   model_name = 'hkunlp/instructor-large'
)

vector_file_path = "faiss_index"

def create_vector_db():
    loader = CSVLoader(file_path = "codebasic_faq.csv", source_column = "Question")
    data = loader.load()
    vector_db = FAISS.from_documents(documents = data, embedding = embeddings)
    vector_db.save_local(vector_file_path)

def get_qa_chain():
    vector_db = FAISS.load_local(vector_file_path, embeddings)

    retriever = vector_db.as_retriever(score_threshold = 0.7)
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context","question"])
    chain_type_kwargs = {"prompt":prompt}


    chain = RetrievalQA.from_chain_type(llm = llm,
                     chain_type = 'stuff',
                     retriever = retriever,
                     input_key = "query",
                     return_source_documents = True,
                     chain_type_kwargs = chain_type_kwargs)

    return chain


if __name__ == "__main__":
    chain = get_qa_chain()

    print(chain("do you provide internship"))
