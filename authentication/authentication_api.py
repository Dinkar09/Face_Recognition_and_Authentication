"""
authentication_api.py

Microservice 3 - Face Authentication (async version).

Exposes a FastAPI endpoint that receives a 512D face embedding,
searches identity_embedding_table for the closest match using
pgvector cosine distance, and returns whether authentication was
Successful or Unsuccessful along with the matched person's name.

Uses an async SQLAlchemy engine (asyncpg driver) so the API can
handle multiple concurrent authentication requests without one
request blocking another on database I/O.
"""

import os
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from sqlalchemy import URL, MetaData, Table, Column, Integer, String, select
from sqlalchemy.ext.asyncio import create_async_engine
from pgvector.sqlalchemy import Vector

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(_CURRENT_DIR, ".env"))

EMBEDDING_DIMENSIONS = 512
DISTANCE_THRESHOLD = float(os.getenv("AUTH_DISTANCE_THRESHOLD", "0.30"))

STATUS_SUCCESSFUL = "Successful"
STATUS_UNSUCCESSFUL = "Unsuccessful"


class DatabaseConfig:
    def __init__(self):
        self.host = os.getenv("DB_HOST", "localhost")
        self.port = int(os.getenv("DB_PORT", "5432"))
        self.database_name = os.getenv("DB_NAME", "IdentityDB")
        self.username = os.getenv("DB_USER", "postgres")
        self.password = os.getenv("DB_PASSWORD", "")

    def build_connection_url(self):
        return URL.create(
            drivername="postgresql+asyncpg",   # was postgresql+psycopg2
            username=self.username,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database_name,
        )


_engine = create_async_engine(DatabaseConfig().build_connection_url(), echo=False)
_metadata = MetaData()

identity_embedding_table = Table(
    "identity_embedding_table",
    _metadata,
    Column("person_unique_id", Integer, primary_key=True),
    Column("person_name", String(150), nullable=False, unique=True),
    Column("facial_data", Vector(EMBEDDING_DIMENSIONS), nullable=False),
)


class FaceMatchService:
    def __init__(self, engine, distance_threshold: float = DISTANCE_THRESHOLD):
        self.engine = engine
        self.distance_threshold = distance_threshold

    async def find_best_match(self, query_embedding: list) -> Optional[dict]:
        distance_column = identity_embedding_table.c.facial_data.cosine_distance(query_embedding).label("distance")

        statement = (
            select(
                identity_embedding_table.c.person_unique_id,
                identity_embedding_table.c.person_name,
                distance_column,
            )
            .order_by(distance_column.asc())
            .limit(1)
        )

        async with self.engine.connect() as connection:
            result = await connection.execute(statement)
            best_row = result.fetchone()

        if best_row is None or best_row.distance > self.distance_threshold:
            return None

        return {
            "person_unique_id": best_row.person_unique_id,
            "person_name": best_row.person_name,
            "distance": float(best_row.distance),
        }


face_match_service = FaceMatchService(_engine)


class FaceEmbeddingRequest(BaseModel):
    embedding: List[float]

    @field_validator("embedding")
    @classmethod
    def validate_embedding_length(cls, value: List[float]) -> List[float]:
        if len(value) != EMBEDDING_DIMENSIONS:
            raise ValueError(f"embedding must have exactly {EMBEDDING_DIMENSIONS} dimensions, got {len(value)}.")
        return value


class AuthenticationResponse(BaseModel):
    status: str
    person_unique_id: Optional[int] = None
    person_name: Optional[str] = None
    distance: Optional[float] = None
    message: str


app = FastAPI(title="Face Authentication API")


@app.post("/authenticate", response_model=AuthenticationResponse)
async def authenticate_face(request: FaceEmbeddingRequest) -> AuthenticationResponse:
    match = await face_match_service.find_best_match(request.embedding)

    if match is None:
        return AuthenticationResponse(
            status=STATUS_UNSUCCESSFUL,
            message="Person not authenticated. Try again.",
        )

    return AuthenticationResponse(
        status=STATUS_SUCCESSFUL,
        person_unique_id=match["person_unique_id"],
        person_name=match["person_name"],
        distance=match["distance"],
        message=f"Person name: {match['person_name']} detected successfully.",
    )


@app.get("/health")
async def health_check() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    API_HOST = os.getenv("API_HOST", "127.0.0.1")
    API_PORT = int(os.getenv("API_PORT", "8000"))

    uvicorn.run(app, host=API_HOST, port=API_PORT)