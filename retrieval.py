from pydantic import BaseModel
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

COLLECTION_NAME = "northstar_hybrid"
TOP_K = 3
MIN_SCORE = 0.30


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str
    position: int


class SearchHit(Chunk):
    score: float


def chunk_documents(documents, max_words=90, overlap_words=15):
    if max_words <= 0 or not 0 <= overlap_words < max_words:
        raise ValueError(
            "Use max_words > 0 and 0 <= overlap_words < max_words."
        )

    chunks = []
    step = max_words - overlap_words

    for doc_id, raw_text in documents.items():
        lines = raw_text.strip().splitlines()

        if not lines or not lines[0].startswith("# "):
            raise ValueError(f"{doc_id}: missing Markdown title.")

        title = lines[0].removeprefix("# ").strip()
        body = "\n".join(lines[1:]).strip()

        if not title or not body:
            raise ValueError(f"{doc_id}: empty title or body.")

        position = 0

        for paragraph in body.split("\n\n"):
            words = paragraph.split()

            for start in range(0, len(words), step):
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}-{position}",
                    doc_id=doc_id,
                    title=title,
                    text=" ".join(words[start:start + max_words]),
                    position=position,
                ))

                position += 1

                # Das Absatzende ist erreicht: kein weiteres Fenster nötig
                if start + max_words >= len(words):
                    break

    if not chunks:
        raise ValueError("No chunks found.")

    return chunks


class Retriever:
    def __init__(self, documents):
        self.chunks = chunk_documents(documents)

        self.model = SentenceTransformer(
            EMBEDDING_MODEL,
            revision=EMBEDDING_REVISION,
            device="cpu",
        )

        vectors = self.model.encode(
            [
                f"{chunk.title}\n{chunk.text}"
                for chunk in self.chunks
            ],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        points = [
            models.PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload=chunk.model_dump(mode="json"),
            )
            for point_id, (chunk, vector) in enumerate(
                zip(self.chunks, vectors, strict=True)
            )
        ]

        self.client = QdrantClient(":memory:")

        try:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=vectors.shape[1],
                    distance=models.Distance.COSINE,
                ),
            )

            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True,
            )

        except Exception:
            self.client.close()
            raise

    def search(self, question):
        if not question.strip():
            raise ValueError("Question must not be empty.")

        vector = self.model.encode(
            [question.strip()],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]

        points = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector.tolist(),
            limit=TOP_K,
            with_payload=True,
            with_vectors=False,
        ).points

        return [
            SearchHit.model_validate({
                **(point.payload or {}),
                "score": point.score,
            })
            for point in points
        ]

    def close(self):
        self.client.close()


def select_evidence(hits):
    return [
        hit
        for hit in hits
        if hit.score >= MIN_SCORE
    ]