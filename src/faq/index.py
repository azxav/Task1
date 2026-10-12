from qdrant_client import QdrantClient, models

from .search.answer_groups import ScoredQuestion


class VectorIndex:
    def __init__(self, url: str, *, alias: str = "faq_questions"):
        self.client = QdrantClient(url=url, timeout=10)
        self.alias = alias

    def generation(self):
        return next(
            (
                alias.collection_name
                for alias in self.client.get_aliases().aliases
                if alias.alias_name == self.alias
            ),
            None,
        )

    def create_generation(self, name, questions, vectors):
        # Existing generation names can only be reused with exactly the same manifest;
        # production importer uses random names, test fixture explicitly recreates its own.
        if self.client.collection_exists(name):
            self.client.delete_collection(name)
        self.client.create_collection(
            name, vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)
        )
        if len(questions) != len(vectors):
            raise ValueError("Questions/vectors length mismatch")
        for start in range(0, len(questions), 128):
            points = [
                models.PointStruct(
                    id=q.id,
                    vector=v.tolist(),
                    payload={"question_id": q.id, "answer_id": q.answer_id, "generation": name},
                )
                for q, v in zip(questions[start : start + 128], vectors[start : start + 128], strict=True)
            ]
            self.client.upsert(name, points=points, wait=True)

    def publish(self, name):
        operations = []
        if self.generation() is not None:
            operations.append(
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=self.alias))
            )
        operations.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(collection_name=name, alias_name=self.alias)
            )
        )
        self.client.update_collection_aliases(change_aliases_operations=operations)

    def search(self, vector, *, expected_generation, limit=30):
        if self.generation() != expected_generation:
            raise RuntimeError("Database/index generation mismatch")
        points = self.client.query_points(
            collection_name=self.alias,
            query=vector.tolist(),
            limit=limit,
            with_payload=True,
            search_params=models.SearchParams(exact=True),
        ).points
        if any(point.payload.get("generation") != expected_generation for point in points):
            raise RuntimeError("Point generation mismatch")
        return [ScoredQuestion(int(p.id), int(p.payload["answer_id"]), p.score) for p in points]
