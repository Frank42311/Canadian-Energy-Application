from pydantic import BaseModel
from typing import List, Optional

class QueryItem(BaseModel):
    """
    A Pydantic model that defines the structure for query items submitted to the API.

    Attributes:
    - query (str): The main text query from the user.
    - sector (str): The sector or category of the query (unused in the given code but may be intended for future use).
    - source (str): The source of data ('Online' for querying OpenAI's model directly or another keyword for local processing).
    - images (Optional[List[str]]): An optional list of base64-encoded images associated with the query.
    """
    query: str
    sector: str
    source: str
    images: Optional[List[str]] = None