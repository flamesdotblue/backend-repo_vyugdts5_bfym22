import os
import datetime
from typing import Any, Dict, List, Optional
from pymongo import MongoClient
from pymongo.collection import Collection

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "appdb")

_client = MongoClient(DATABASE_URL)
db = _client[DATABASE_NAME]


def _get_collection(name: str) -> Collection:
    return db[name]


def create_document(collection_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    now = datetime.datetime.utcnow()
    payload = {**data, "created_at": now, "updated_at": now}
    col = _get_collection(collection_name)
    result = col.insert_one(payload)
    payload["_id"] = str(result.inserted_id)
    return payload


def get_documents(
    collection_name: str, filter_dict: Optional[Dict[str, Any]] = None, limit: int = 50
) -> List[Dict[str, Any]]:
    col = _get_collection(collection_name)
    cursor = col.find(filter_dict or {}).limit(limit)
    docs: List[Dict[str, Any]] = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])  # serialize ObjectId
        docs.append(doc)
    return docs


def update_document(collection_name: str, filter_dict: Dict[str, Any], update: Dict[str, Any]) -> int:
    col = _get_collection(collection_name)
    update_payload = {"$set": {**update, "updated_at": datetime.datetime.utcnow()}}
    res = col.update_one(filter_dict, update_payload, upsert=False)
    return res.modified_count
