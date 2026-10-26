import hashlib
import uuid


def new_doc_id() -> str:
    return uuid.uuid4().hex


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
