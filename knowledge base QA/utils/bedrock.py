import boto3
import os
import hashlib
from botocore.config import Config

_cache = {}

def retrieve_from_kb(question: str, num_results: int = 5) -> str:
    cache_key = hashlib.md5(question.strip().lower().encode()).hexdigest()
    if cache_key in _cache:
        print("Cache hit — skipping AWS call")
        return _cache[cache_key]

    config = Config(
        retries={"max_attempts": 10, "mode": "adaptive"},
        read_timeout=120,
        connect_timeout=15
    )

    bedrock = boto3.client(
        "bedrock-agent-runtime",
        region_name=os.getenv("AWS_REGION", "ap-south-1"),
        config=config
    )
    kb_id = os.getenv("KNOWLEDGE_BASE_ID")

    # Use RetrieveAndGenerate — separate quota from Retrieve
    response = bedrock.retrieve_and_generate(
        input={"text": question},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": kb_id,
                "modelArn": "arn:aws:bedrock:ap-south-1::foundation-model/amazon.titan-text-express-v1",
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {"numberOfResults": num_results}
                }
            }
        }
    )

    # Extract the generated answer and source chunks
    answer = response.get("output", {}).get("text", "")
    citations = response.get("citations", [])
    
    chunks = []
    for citation in citations:
        for ref in citation.get("retrievedReferences", []):
            text = ref.get("content", {}).get("text", "")
            if text:
                chunks.append(text)

    context = "\n\n".join(chunks)
    _cache[cache_key] = context

    return context, answer  # returns both context AND a pre-generated answer