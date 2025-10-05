from loguru import logger
from qdrant_client import QdrantClient, models


class QdrantConnectionError(Exception):
    pass


class QdrantConnector:
    def __init__(self, url: str) -> None:
        self.client = QdrantClient(url=url)
        self._test_connection()

    def _test_connection(self) -> None:
        try:
            info = self.client.info()
            logger.info(f"Qdrant connection successful!\n{info}")
        except Exception as e:
            logger.error(f"Qdrant connection failed:\n{e}")
            raise QdrantConnectionError(f"Qdrant connection failed: {e}")

    def ensure_collection_exists(
        self, collection_name: str, vector_dim: int = 128
    ) -> None:
        if not self.client.collection_exists(collection_name=collection_name):
            logger.info(
                f"Collection '{collection_name}' does not exist. Creating collection... "
            )
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=128,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                ),
            )
        else:
            logger.info(f"Collection '{collection_name}' exists.")

    def upsert_points(self, points: list[models.PointStruct], collection_name: str):
        self.client.upsert(collection_name=collection_name, wait=True, points=points)
