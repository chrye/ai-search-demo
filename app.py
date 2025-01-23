# Import required libraries
import os
import json
from xml.dom.minidom import Document  
import wget
import pandas as pd
import zipfile
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.core.credentials import AzureKeyCredential  
from azure.core.exceptions import HttpResponseError
from azure.search.documents import SearchClient, SearchIndexingBufferedSender  
from azure.search.documents.indexes import SearchIndexClient  
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryType,
    VectorizedQuery,
)
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    HnswParameters,
    SearchField,
    SearchableField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchAlgorithmKind,
    VectorSearchAlgorithmMetric,
    VectorSearchProfile,
)

#Configure Settings
#region 

print("configuring settings...")

load_dotenv()

# Configure OpenAI settings
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(
    credential, "https://cognitiveservices.azure.com/.default"
)


# Set this flag to True if you are using Azure Active Directory
use_aad_for_aoai = False 

if use_aad_for_aoai:
    # Use Azure Active Directory (AAD) authentication
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_version=api_version,
        azure_ad_token_provider=token_provider,
    )
else:
    # Use API key authentication
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )


# Configure Azure AI Search Vector Store settings

search_service_endpoint: str = os.getenv("AZURE_SEARCH_ENDPOINT")
search_service_api_key: str = os.getenv("AZURE_SEARCH_ADMIN_KEY")
index_name: str = os.getenv("AZURE_SEARCH_INDEX_NAME")

# Set this flag to True if you are using Azure Active Directory
use_aad_for_search = False  

if use_aad_for_search:
    # Use Azure Active Directory (AAD) authentication
    credential = DefaultAzureCredential()
else:
    # Use API key authentication
    credential = AzureKeyCredential(search_service_api_key)

# Initialize the SearchClient with the selected authentication method
search_client = SearchClient(
    endpoint=search_service_endpoint, index_name=index_name, credential=credential
)

#endregion

print("downloading and extracting data...")

embeddings_url = "https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip"

# The file is ~700 MB so this will take some time
wget.download(embeddings_url)

print("...zip file downloaded")

with zipfile.ZipFile("vector_database_wikipedia_articles_embedded.zip", "r") as zip_ref:
    zip_ref.extractall("data")

print("...zip file extracted")

# Read the CSV file
article_df = pd.read_csv("data/vector_database_wikipedia_articles_embedded.csv")
# Defining the columns to read
#usecols = ["id", "url", "title", "text", "vector_id", "title_vector"]
# Read data with subset of columns
#article_df = pd.read_csv("data/vector_database_wikipedia_articles_embedded.csv", usecols=usecols)

print("...csv content loaded")

# Read vectors from strings back into a list using json.loads
article_df["title_vector"] = article_df.title_vector.apply(json.loads)
#print(article_df["title_vector"])
article_df["content_vector"] = article_df.content_vector.apply(json.loads)
article_df["vector_id"] = article_df["vector_id"].apply(str)
#print(article_df["vector_id"])

#article_df.head()
#article_df.info()
#print(article_df.head())

print("...vectors loaded from strings into a list")

# Create an index
print("creating index...")

# Initialize the SearchIndexClient
index_client = SearchIndexClient(
    endpoint=search_service_endpoint, credential=credential
)

# Define the fields for the index
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String),
    SimpleField(name="vector_id", type=SearchFieldDataType.String, key=True),
    SimpleField(name="url", type=SearchFieldDataType.String),
    SearchableField(name="title", type=SearchFieldDataType.String),
    SearchableField(name="text", type=SearchFieldDataType.String),
    SearchField(
        name="title_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        vector_search_dimensions=1536,
        vector_search_profile_name="my-vector-config",
    ),
    SearchField(
        name="content_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        vector_search_dimensions=1536,
        vector_search_profile_name="my-vector-config",
    ),
]

# Configure the vector search configuration
vector_search = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(
            name="my-hnsw",
            kind=VectorSearchAlgorithmKind.HNSW,
            parameters=HnswParameters(
                m=4,
                ef_construction=400,
                ef_search=500,
                metric=VectorSearchAlgorithmMetric.COSINE,
            ),
        )
    ],
    profiles=[
        VectorSearchProfile(
            name="my-vector-config",
            algorithm_configuration_name="my-hnsw",
        )
    ],
)

print("...vector search configured")

# Configure the semantic search configuration
semantic_search = SemanticSearch(
    configurations=[
        SemanticConfiguration(
            name="my-semantic-config",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="title"),
                keywords_fields=[SemanticField(field_name="url")],
                content_fields=[SemanticField(field_name="text")],
            ),
        )
    ]
)

print("...semantic search configured")

# Create the search index with the vector search and semantic search configurations
index = SearchIndex(
    name=index_name,
    fields=fields,
    vector_search=vector_search,
    semantic_search=semantic_search,
)

# Create or update the index
result = index_client.create_or_update_index(index)
print(f"...Index <{result.name}> created/updated")


# Uploading Data to Azure AI Search Index
print("uploading data to AI search index...")

# Convert the 'id' and 'vector_id' columns to string so one of them can serve as our key field
article_df["id"] = article_df["id"].astype(str)
article_df["vector_id"] = article_df["vector_id"].astype(str)
# Convert the DataFrame to a list of dictionaries
documents = article_df.to_dict(orient="records")

print(f"   total # of records: {len(documents)}. uploading...")

# Option 1 - Create a SearchIndexingBufferedSender and upload all documents in a single call 
# batch_client = SearchIndexingBufferedSender(
#     search_service_endpoint, index_name, credential
# )

# try:
#     # Add upload actions for all documents in a single call
#     batch_client.upload_documents(documents=documents)

#     # Manually flush to send any remaining documents in the buffer
#     batch_client.flush()
# except HttpResponseError as e:
#     print(f"An error occurred: {e}")
# finally:
#     # Clean up resources
#     batch_client.close()

# Option 2 - Use search_client to upload one doc at a time
for doc in documents:
    try:
        search_client.upload_documents(documents=[doc])
        if int(doc['vector_id']) % 1000 == 0:
            print(f"...uploaded document with id: {doc['id']}, {doc['vector_id']}")

    except HttpResponseError as hre:
        print(f"An error occurred while uploading document with id {doc['id']}: {hre}")
    except Exception as e:
        print(f"Exception occurred while uploading document with id {doc['id']}: {e}")

#print(f"Uploaded {len(documents)} documents in total")

# Validate the document count 
document_count = search_client.get_document_count()
print(f"...Document count in the index: {document_count}")

# Compare with the total number of documents in your DataFrame
total_documents = len(article_df)
print(f"...Total number of documents in the DataFrame: {total_documents}")

if document_count == total_documents:
    print("...All documents were uploaded correctly.")
else:
    print("WARNING: There is a discrepancy in the document count. Please check the logs for any errors.")


# Example function to generate document embedding
def generate_embeddings(text, model):
    # Generate embeddings for the provided text using the specified model
    embeddings_response = client.embeddings.create(model=model, input=text)
    # Extract the embedding data from the response
    embedding = embeddings_response.data[0].embedding
    return embedding

first_document_content = documents[0]["text"]
print(f"...content: {first_document_content[:100]}")

content_vector = generate_embeddings(first_document_content, deployment)
print("...content vector generated")


print("\n\n==============================\n\n")


# Perform a vector similarity search

# Pure Vector Search
query = "modern art in Europe"

print("---")
print(f"Search Test - Vector similarity search: {query}")
print("---\n")

search_client = SearchClient(search_service_endpoint, index_name, credential)  
vector_query = VectorizedQuery(vector=generate_embeddings(query, deployment), k_nearest_neighbors=3, fields="content_vector")
  
results = search_client.search(  
    search_text=None,  
    vector_queries= [vector_query], 
    select=["title", "text", "url"] 
)
  
for result in results:  
    print(f"Title: {result['title']}")  
    print(f"Score: {result['@search.score']}")  
    print(f"URL: {result['url']}\n")  

# Perform a Hybrid Search

# Hybrid Search
query = "Famous battles in Scottish history"  

print("---")
print(f"Search Test - Hybrid search: {query}")
print("---\n")

search_client = SearchClient(search_service_endpoint, index_name, credential)  
vector_query = VectorizedQuery(vector=generate_embeddings(query, deployment), k_nearest_neighbors=3, fields="content_vector")
  
results = search_client.search(  
    search_text=query,  
    vector_queries= [vector_query], 
    select=["title", "text", "url"],
    top=3
)
  
for result in results:  
    print(f"Title: {result['title']}")  
    print(f"Score: {result['@search.score']}")  
    print(f"URL: {result['url']}\n")  


# Perform a Hybrid Search with Reranking (powered by Bing)

# Semantic Hybrid Search
query = "What were the key technological advancements during the Industrial Revolution?"

print("---")
print(f"Search Test - Semantic Hybrid search: {query}")
print("---\n")

search_client = SearchClient(search_service_endpoint, index_name, credential)
vector_query = VectorizedQuery(
    vector=generate_embeddings(query, deployment),
    k_nearest_neighbors=3,
    fields="content_vector",
)

results = search_client.search(
    search_text=query,
    vector_queries=[vector_query],
    select=["title", "text", "url"],
    query_type=QueryType.SEMANTIC,
    semantic_configuration_name="my-semantic-config",
    query_caption=QueryCaptionType.EXTRACTIVE,
    query_answer=QueryAnswerType.EXTRACTIVE,
    top=3,
)

semantic_answers = results.get_answers()
for answer in semantic_answers:
    if answer.highlights:
        print(f"Semantic Answer: {answer.highlights}")
    else:
        print(f"Semantic Answer: {answer.text}")
    print(f"Semantic Answer Score: {answer.score}\n")

for result in results:
    print(f"Title: {result['title']}")
    print(f"Reranker Score: {result['@search.reranker_score']}")
    print(f"URL: {result['url']}")
    captions = result["@search.captions"]
    if captions:
        caption = captions[0]
        if caption.highlights:
            print(f"Caption: {caption.highlights}\n")
        else:
            print(f"Caption: {caption.text}\n")


print("---- THE END")
			