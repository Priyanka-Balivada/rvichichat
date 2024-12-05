import subprocess
from pathlib import Path
from typing import Optional, List
from fastapi import Body, FastAPI, Form, HTTPException, Query
from fastapi.responses import HTMLResponse
from llama_index.readers.web import SimpleWebPageReader, BeautifulSoupWebReader, RssReader, WholeSiteReader
from llama_index.core import SummaryIndex
import json
from llama_index.readers.file import (
    DocxReader,
    PDFReader,
    FlatReader,
    HTMLTagReader,
    ImageReader,
    IPYNBReader,
    PptxReader,
    PandasCSVReader,
    PyMuPDFReader,
    XMLReader,
    CSVReader,
)

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.text_embeddings_inference import TextEmbeddingsInference
from llama_index.vector_stores.milvus.utils import get_default_sparse_embedding_function
from pymilvus import connections, utility, FieldSchema, CollectionSchema, DataType, Collection
import os
from pydantic import BaseModel
from typing import Dict, Any
from pymilvus import connections
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore,IndexManagement 
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.core.node_parser import SentenceSplitter
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.schema import NodeWithScore
from pymilvus import connections, utility
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.ollama import Ollama
from typing import List, Optional
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.llms.text_generation_inference import (
    TextGenerationInference,
)

import uuid
import redis
import json

app = FastAPI()
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allows GET, POST, OPTIONS, etc.
    allow_headers=["*"],  # Allows all headers
)
@app.options("/read")
async def preflight():
    """Handle OPTIONS requests for CORS preflight."""
    return JSONResponse(status_code=200, content={"message": "Preflight request successful"})

# Connect to Redis
redis_client =redis.Redis(host="localhost", port=6379, db=0)
task_data_name=""

class Task(BaseModel):
    description: str
    status: str

class ReadRequest(BaseModel):
    #content ingestion
    format: str = Form(...),
    url_web: Optional[str] = Form(None),
    url_html: Optional[str] = Form(None),
    url_soup: Optional[str] = Form(None),
    url_whole: Optional[str] = Form(None),
    url_file: Optional[str] = Form(None),
    tag: Optional[str] = Form("section"),
    ignore_no_id: Optional[bool] = Form(True),
    prefix: Optional[str] = Form(None),
    max_depth: Optional[int] = Form(10)

    #task management configurations
    task_data_management_collection: str ="task_data"
    taskHost: str="localhost"
    taskPort: int=6379

    #embedding model configurations
    modelName: str = "BAAI/bge-small-en-v1.5"
    modelBaseUrl: str = "http://127.0.0.1:8081"
    # modelTextInstruction: str = None
    # modelQueryInstruction: str = None
    modelEmbedBatchSize: int = 10
    modelTimeout: float = 60.0
    modelTruncateText: bool = True
    # callback_manager: Optional[CallbackManager] = None
    # auth_token: Optional[Union[str, Callable[[str], str]]] = None

    #docstore configurations
    docHost: str = "localhost"
    docPort: int = 6379
    docCollectionName: str = "document_store"

    #cache configurations
    cacheHost: str = "127.0.0.1"
    cachePort: int = 6379
    cacheCollectionName: str = "cache"

    #vectorstore configurations
    milvusURI: str = "http://localhost:19530"
    milvusDimension: int= 1024
    milvusOverwrite: bool= False
    milvusCollectionName: str= "app_milvus_db"
    milvusConsistencyLevel: str= "Eventually"
    milvusEnableSparse: bool= True
    milvusSimilarityMetric: str= "IP"
    milvusToken: str = ""
    milvusEmbeddingField: str = "embedding"
    milvusDocIdField: str = "doc_id"
    milvusTextKey: str = "text"
    milvusSparseEmbeddingField: str = "sparse_embedding"
    milvusIndexManagement: IndexManagement = IndexManagement.CREATE_IF_NOT_EXISTS
    milvusBatchSize: int = 100
    milvusSparseEmbeddingFunction: Any = get_default_sparse_embedding_function()

    # output_fields: List[str] = Field(default_factory=list)
    # index_config: Optional[dict]
    # search_config: Optional[dict]
    # collection_properties: Optional[dict]
    # hybrid_ranker: str
    # hybrid_ranker_params: dict = {}
    # scalar_field_names: Optional[List[str]]
    # scalar_field_types: Optional[List[DataType]]

class QueryRequest(BaseModel):
    #query configurations
    query: str = "Give summary of Documents"

    #LLM Model configurations
    baseLLMUrl: str = "http://localhost:11434"
    selectedModel: str = ""

    #embedding model configurations
    modelName: str = "BAAI/bge-small-en-v1.5"
    modelBaseUrl: str = "http://127.0.0.1:8081"
    # modelTextInstruction: str = None
    # modelQueryInstruction: str = None
    modelEmbedBatchSize: int = 10
    modelTimeout: float = 60.0
    modelTruncateText: bool = True
    # callback_manager: Optional[CallbackManager] = None
    # auth_token: Optional[Union[str, Callable[[str], str]]] = None

    #vectorstore configurations
    milvusURI: str = "http://localhost:19530"
    milvusDimension: int= 1024
    milvusOverwrite: bool= False
    milvusCollectionName: str= "app_milvus_db"
    milvusConsistencyLevel: str= "Eventually"
    milvusEnableSparse: bool= True
    milvusSimilarityMetric: str= "IP"
    milvusToken: str = ""
    milvusEmbeddingField: str = "embedding"
    milvusDocIdField: str = "doc_id"
    milvusTextKey: str = "text"
    milvusSparseEmbeddingField: str = "sparse_embedding"
    milvusIndexManagement: IndexManagement = IndexManagement.CREATE_IF_NOT_EXISTS
    milvusBatchSize: int = 100
    milvusSparseEmbeddingFunction: Any = get_default_sparse_embedding_function()

    # output_fields: List[str] = Field(default_factory=list)
    # index_config: Optional[dict]
    # search_config: Optional[dict]
    # collection_properties: Optional[dict]
    # hybrid_ranker: str
    # hybrid_ranker_params: dict = {}
    # scalar_field_names: Optional[List[str]]
    # scalar_field_types: Optional[List[DataType]]

class TaskRequest(BaseModel):
    #task management configurations
    task_data_management_collection: str ="task_data"
    taskHost: str="localhost"
    taskPort: int=6379

class TaskRequestById(BaseModel):
    #task management configurations
    task_data_management_collection: str ="task_data"
    taskHost: str="localhost"
    taskPort: int=6379
    task_id: str

class VectorDBRetriever(BaseRetriever):
        """Retriever over a milvus vector store."""

        def __init__(
            self,
            vector_store: MilvusVectorStore,
            embed_model: Any,
            query_mode: str = "default",
            similarity_top_k: int = 2,
        ) -> None:
            """Init params."""
            self._vector_store = vector_store
            self._embed_model = embed_model
            self._query_mode = query_mode
            self._similarity_top_k = similarity_top_k
            super().__init__()

        def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
            """Retrieve."""
            query_embedding = embed_model.get_query_embedding(
                query_bundle.query_str
            )
            vector_store_query = VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=self._similarity_top_k,
                mode=self._query_mode,
            )
            query_result = vector_store.query(vector_store_query)

            nodes_with_scores = []
            for index, node in enumerate(query_result.nodes):
                score: Optional[float] = None
                if query_result.similarities is not None:
                    score = query_result.similarities[index]
                nodes_with_scores.append(NodeWithScore(node=node, score=score))

            return nodes_with_scores


def create_task(description: str) -> str:
    task_id = str(uuid.uuid4())
    task_data = {"description": description, "status": "in process"}
    redis_client.hset(task_data_name, task_id, json.dumps(task_data))
    return task_id

def update_task_status(task_id: str, status: str):
    if redis_client.hexists(task_data_name, task_id):
        task_data = json.loads(redis_client.hget(task_data_name, task_id))
        task_data["status"] = status
        redis_client.hset(task_data_name, task_id, json.dumps(task_data))

@app.get("/tasks", response_model=Dict[str, Task])
def get_all_tasks(task_data_name: str = Query("task_data", description="Name of the Redis hash for tasks"),
    host: str = Query("localhost", description="Redis server host"),
    port: int = Query(6379, description="Redis server port")):

    print(task_data_name)
    redis_client = redis.Redis(host=host, port=port, db=0)
    print(redis_client)
    task_data = redis_client.hgetall(task_data_name)
    tasks = {task_id.decode('utf-8'): json.loads(task_info) for task_id, task_info in task_data.items()}
    return tasks

@app.get("/tasks/{task_id}", response_model=Task)
def get_task(task_id: str,
    task_data_name: str = Query("task_data", description="Name of the Redis hash for tasks"),
    host: str = Query("localhost", description="Redis server host"),
    port: int = Query(6379, description="Redis server port")):

    redis_client = redis.Redis(host=host, port=port, db=0)
    if redis_client.hexists(task_data_name, task_id):
        task_info = json.loads(redis_client.hget(task_data_name, task_id))
        return task_info
    else:
        raise HTTPException(status_code=404, detail="Task not found")


@app.post("/read")
def read_data(request: ReadRequest = Body(ReadRequest())):
    global task_data_name
    global redis_client
    documents=None
    
    if request.format == "web":
        if request.url_web is None:
            raise HTTPException(status_code=400, detail="URL must be provided for web reader")
        documents = SimpleWebPageReader(html_to_text=True).load_data([request.url_web])
        
    elif request.format == "html_tags":
        if request.url_html is None:
            raise HTTPException(status_code=400, detail="URL must be provided for HTML tag reader")
        if not os.path.exists(request.url_html):
            raise HTTPException(status_code=400, detail="File not found")
        with open(request.url_html, "r", encoding="utf-8") as f:
            content = f.read()
        reader = HTMLTagReader(tag=request.tag, ignore_no_id=request.ignore_no_id)
        documents = reader.load_data(content)
    
    elif request.format == "beautiful_soup":
        if request.url_soup is None:
            raise HTTPException(status_code=400, detail="URL must be provided for BeautifulSoup reader")
        loader = BeautifulSoupWebReader()
        documents = loader.load_data(urls=[request.url_soup])
    
    elif request.format == "rss":
        reader = RssReader()
        documents = reader.load_data(
            [
                "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
                "https://roelofjanelsinga.com/atom.xml",
            ]
        )
    
    elif request.format == "whole_site":
        if request.url_whole is None or request.prefix is None:
            raise HTTPException(status_code=400, detail="Base URL and prefix must be provided for Whole Site Reader")
        scraper = WholeSiteReader(prefix=request.prefix, max_depth=request.max_depth)
        documents = scraper.load_data(base_url=request.url_whole)
    
    elif request.format in ["pdf", "docx", "txt", "image", "ipynb", "pptx", "csv", "xml"]:
        if request.url_file is None:
            raise HTTPException(status_code=400, detail="URL must be provided for the selected format")
        if not os.path.exists(request.url_file):
            raise HTTPException(status_code=400, detail="File not found")
        
        parser_map = {
            "pdf": PyMuPDFReader(),
            "docx": DocxReader(),
            "txt": FlatReader(),
            "image": ImageReader(),
            "ipynb": IPYNBReader(),
            "pptx": PptxReader(),
            "csv": CSVReader(),
            "xml": XMLReader()
        }
        
        parser = parser_map.get(request.format)
        documents = parser.load_data(request.url_file)

    elif request.format == "dir":
        print(Path(request.url_file))
        print(request.url_file)
        reader = SimpleDirectoryReader(input_dir=Path(request.url_file), recursive=True, required_exts=[".txt", ".pdf"])
        documents = reader.load_data()

    parameters = {
        'modelName': request.modelName,
        'docHost': request.docHost,
        'docPort': request.docPort,
        'docCollectionName': request.docCollectionName,
        'cacheHost': request.cacheHost,
        'cachePort': request.cachePort,
        'cacheCollectionName': request.cacheCollectionName,
        'milvusURI': request.milvusURI,
        'milvusDimension': request.milvusDimension,
        'milvusOverwrite': request.milvusOverwrite,
        'milvusCollectionName': request.milvusCollectionName,
        # 'milvusConsistencyLevel': request.milvusConsistencyLevel,
        'milvusEnableSparse': request.milvusEnableSparse,
        # 'milvusSimilarityMetric': request.milvusSimilarityMetric,
        # 'milvusToken': request.milvusToken,
        # 'milvusEmbeddingField': request.milvusEmbeddingField,
        # 'milvusDocIdField': request.milvusDocIdField,
        # 'milvusTextKey': request.milvusTextKey,
        # 'milvusSparseEmbeddingField': request.milvusSparseEmbeddingField,
        # 'milvusIndexManagement': request.milvusIndexManagement,
        'milvusBatchSize': request.milvusBatchSize,
        # 'milvusSparseEmbeddingFunction': request.milvusSparseEmbeddingFunction,
        'modelBaseUrl': request.modelBaseUrl,
        # 'modelTextInstruction': request.modelTextInstruction,
        # 'modelQueryInstruction': request.modelQueryInstruction,
        'modelEmbedBatchSize': request.modelEmbedBatchSize,
        'modelTimeout': request.modelTimeout,
        # 'modelTruncateText': request.modelTruncateText,
        # 'taskHost': request.taskHost,
        # 'taskPort': request.taskPort
    }
    
    task_data_name = request.task_data_management_collection

    redis_client = redis.Redis(host=request.taskHost, port=request.taskPort, db=0)

    task_id = create_task("Loading documents")
    nodes = []  
    try:
        nodes = ingestion(documents, task_id, parameters)
        update_task_status(task_id, "completed")
    except Exception as e:
        update_task_status(task_id, f"failed: {str(e)}")
        print(e)
    
    return nodes
     

@app.post("/generate")
async def generate(request: QueryRequest = Body(QueryRequest())):
    """
    Query endpoint for generating responses based on the input query.
    """
    parameters = {
        'modelName': request.modelName,
        'milvusURI': request.milvusURI,
        'milvusDimension': request.milvusDimension,
        'milvusOverwrite': request.milvusOverwrite,
        'milvusCollectionName': request.milvusCollectionName,
        'milvusConsistencyLevel': request.milvusConsistencyLevel,
        'milvusEnableSparse': request.milvusEnableSparse,
        'milvusSimilarityMetric': request.milvusSimilarityMetric,
        'milvusToken': request.milvusToken,
        'milvusEmbeddingField': request.milvusEmbeddingField,
        'milvusDocIdField': request.milvusDocIdField,
        'milvusTextKey': request.milvusTextKey,
        'milvusSparseEmbeddingField': request.milvusSparseEmbeddingField,
        'milvusIndexManagement': request.milvusIndexManagement,
        'milvusBatchSize': request.milvusBatchSize,
        'milvusSparseEmbeddingFunction': request.milvusSparseEmbeddingFunction,
        'modelBaseUrl': request.modelBaseUrl,
        # 'modelTextInstruction': request.modelTextInstruction,
        # 'modelQueryInstruction': request.modelQueryInstruction,
        'modelEmbedBatchSize': request.modelEmbedBatchSize,
        'modelTimeout': request.modelTimeout,
        'modelTruncateText': request.modelTruncateText,

        'query': request.query,

        'baseLLMUrl': request.baseLLMUrl,
        'selectedModel': request.selectedModel
    }

    global embed_model, vector_store
    
    try:
        embed_model=TextEmbeddingsInference(
                model_name=parameters['modelName'],
                base_url=parameters['modelBaseUrl'],
                # text_instruction=parameters['modelTextInstruction'],
                # query_instruction=parameters['modelQueryInstruction'],
                embed_batch_size=parameters['modelEmbedBatchSize'],
                timeout= parameters['modelTimeout'],
                # truncate_text=parameters['modelTruncateText'],
                # callback_manager: Optional[CallbackManager] = None,
                # auth_token: Optional[Union[str, Callable[[str], str]]] = None,
            )

        # Connect to Milvus server
        connections.connect("default", uri=parameters["milvusURI"])  # Replace with your Milvus URI

        # Collection name to check
        collection_name = parameters["milvusCollectionName"]

        # Check if the collection exists
        try:
            exists = utility.has_collection(collection_name)
            if exists:
                print(f"The collection '{collection_name}' exists.")
            else:
                print(f"The collection '{collection_name}' does not exist.")
                return {"status": "error", "response": f"Collection '{collection_name}' does not exist in Milvus."}
        except Exception as e:
            print(f"An error occurred: {e}")

        # vector_store=MilvusVectorStore(
        #         uri=parameters["milvusURI"],  
        #         overwrite=parameters["milvusOverwrite"],
        #         collection_name=parameters["milvusCollectionName"],
        #         enable_sparse=True,
        #     )
        

        # Step 1: Import and Initialize Embedding Model
        print("Importing and Initializing Embedding Model...")
        # embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")  # Replace with your model
        print("Embedding Model Loaded")

        # Step 2: Define Query String and Get Embeddings
        # query_str = "What is scaling factor"
        query_embedding = embed_model.get_query_embedding(parameters["query"])

        # Step 3: Define Query Mode as Hybrid
        query_mode = "hybrid"

        # Step 4: Configure MilvusVectorStore for Hybrid Search
        vector_store = MilvusVectorStore(
            uri=parameters["milvusURI"],  # Adjust as needed
            dim=parameters["milvusDimension"],  # Adjust dimension based on your embedding model
            overwrite=False,
            collection_name=parameters["milvusCollectionName"],
            consistency_level="Eventually",
            enable_sparse=True,  # Enable sparse search for hybrid functionality
            # sparse_threshold=0.01,  # Add threshold for sparse search
            # sparse_model_name="BAAI/bge-m3",  # Specify the sparse model
        )

        print("Milvus Vector Store Configured")

        # Step 5: Construct the VectorStoreQuery for Hybrid Search
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            query_str=parameters["query"],  # Add the query string for sparse search
            similarity_top_k=5,  # Number of results to return
            mode=query_mode,  # Set query mode to hybrid
            alpha=0.5,  # Balance between dense and sparse scores
        )

        print("Vector Query Created")

        # Step 6: Execute the Query
        query_result = vector_store.query(vector_store_query)

        # Step 7: Display Retrieved Context
        retrieved_context = ""

        # Step 8: Configure LlamaCPP for Language Generation
        print(parameters["baseLLMUrl"])
        print(parameters["selectedModel"])
        llm = Ollama(base_url=parameters["baseLLMUrl"], model=parameters["selectedModel"], request_timeout=300.0)

        vector_index = VectorStoreIndex.from_vector_store(vector_store,embed_model)
        retriever = VectorIndexRetriever(vector_store=vector_store, embed_model=embed_model, index=vector_index)
        query_engine = RetrieverQueryEngine.from_args(retriever=retriever, llm=llm)

        # Provide the retrieved context to the query engine
        response = query_engine.query(parameters["query"])

        return {"status": "success", "response": str(response)}

    except Exception as e:
        return {"status": "error", "message": str(e)}
    
def ingestion(documents,load_task,parameters):
    task_id = create_task("Running ingestion pipeline")
    try:
        pipeline = IngestionPipeline(
            transformations=[
                # SentenceSplitter(chunk_size=100, chunk_overlap=0),
                SentenceSplitter(chunk_size=512, chunk_overlap=10),
                TokenTextSplitter(),
                # HuggingFaceEmbedding(model_name=parameters['modelName']),
                TextEmbeddingsInference(
                model_name=parameters['modelName'],
                base_url=parameters['modelBaseUrl'],
                # text_instruction=parameters['modelTextInstruction'],
                # query_instruction=parameters['modelQueryInstruction'],
                embed_batch_size=parameters['modelEmbedBatchSize'],
                timeout= parameters['modelTimeout'],
                # truncate_text=parameters['modelTruncateText'],
                # callback_manager: Optional[CallbackManager] = None,
                # auth_token: Optional[Union[str, Callable[[str], str]]] = None,
                ),
            ],
            docstore=RedisDocumentStore.from_host_and_port(
                parameters['docHost'], parameters['docPort'], namespace=parameters['docCollectionName']
            ),
            cache=IngestionCache(
                cache=RedisCache.from_host_and_port(host=parameters['cacheHost'], port=parameters['cachePort']),
                collection=parameters['cacheCollectionName'],
            ), 
            vector_store=MilvusVectorStore(
                uri=parameters["milvusURI"],  
                dim=parameters["milvusDimension"],  
                overwrite=parameters["milvusOverwrite"],
                collection_name=parameters["milvusCollectionName"],
                # consistency_level=parameters["milvusConsistencyLevel"],
                enable_sparse=parameters["milvusEnableSparse"],
                # similarity_metric=parameters["milvusSimilarityMetric"],
                # token=parameters['milvusToken'],
                # embedding_field=parameters['milvusEmbeddingField'],
                # doc_id_field=parameters['milvusDocIdField'],
                # text_key=parameters['milvusTextKey'],
                # sparse_embedding_field=parameters['milvusSparseEmbeddingField'],
                # hybrid_ranker="RRFRanker",
                # hybrid_ranker_params={"k": 60},
                # scalar_field_names=['text','file_size'],
                # scalar_field_types=[DataType.VARCHAR,DataType.INT64],
                batch_size=parameters['milvusBatchSize'],
                # sparse_embedding_function: Any
                # hybrid_ranker: str
                # hybrid_ranker_params: dict = {}
                # index_management = parameters['milvusIndexManagement'],
                # sparse_embedding_function = parameters['milvusSparseEmbeddingFunction'],
            )
        )

        
        nodes = pipeline.run(documents=documents)
        update_task_status(task_id, "completed")

        # print_milvus_contents(parameters["app8_db"])
    except Exception as e:
        update_task_status(task_id, f"failed: {str(e)}")
        print(f"Pipeline Error: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline Error: {e}")
    return {"insert_results": len(nodes),"Pipeline Task ID":task_id,"Load Task ID":load_task}

def print_milvus_contents(collection_name: str):
    try:
        # Connect to Milvus server
        connections.connect("default", uri="http://localhost:19530")
        
        # Check if the collection exists
        if not utility.has_collection(collection_name):
            print(f"Collection '{collection_name}' not found.")
            return

        # Load the collection
        collection = Collection(name=collection_name)

        # Retrieve all data from the collection
        data = collection.query(expr="*", output_fields=["doc_id", "embedding", "text"], limit=10)
        
        print(f"\n--- Contents of Milvus Collection: {collection_name} ---")
        for idx, doc in enumerate(data):
            print(f"\nDocument {idx + 1}:")
            print(f"ID: {doc['doc_id']}")
            print(f"Text Key: {doc['text']}")
            print(f"Embedding (first 10 dims): {doc['embedding'][:10]}")

    except Exception as e:
        print(f"Error querying Milvus: {e}")

@app.get("/getmodels")
def get_ollama_models():
    try:
        # Execute the 'ollama list' command and capture the output
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True)
        
        # Parse the output (assuming the models are listed line by line)
        models = result.stdout.splitlines()
        model_names = [model.split()[0].strip() for model in models if model.strip()]  # Take first word as model name
        print(model_names)
        return model_names
    except subprocess.CalledProcessError as e:
        return {"error": f"Failed to list models: {str(e)}"}

