import hashlib
from contextlib import contextmanager

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .data import Answer

DDL = """
CREATE TABLE IF NOT EXISTS answers (
 id bigint PRIMARY KEY, category_id bigint NOT NULL, topic_id bigint NOT NULL,
 answer text NOT NULL, status text NOT NULL, created_at timestamp NOT NULL, updated_at timestamp NOT NULL);
CREATE TABLE IF NOT EXISTS questions (
 id bigint PRIMARY KEY, answer_id bigint NOT NULL REFERENCES answers(id), question text NOT NULL,
 embedding_text text NOT NULL, is_active boolean NOT NULL, created_at timestamp NOT NULL, updated_at timestamp NOT NULL);
CREATE INDEX IF NOT EXISTS questions_answer_id ON questions(answer_id);
CREATE TABLE IF NOT EXISTS metadata (id integer PRIMARY KEY CHECK(id=1), value jsonb NOT NULL);
"""


class Storage:
    def __init__(self, url: str, *, schema: str = "faq"):
        self.url, self.schema = url, schema
        self.lock_id = int.from_bytes(hashlib.sha256(schema.encode()).digest()[:8], "big", signed=True)

    @contextmanager
    def connection(self):
        with psycopg.connect(self.url, row_factory=dict_row, connect_timeout=3) as connection:
            connection.execute(sql.SQL("SET search_path TO {}").format(sql.Identifier(self.schema)))
            connection.execute("SET statement_timeout TO '10s'")
            yield connection

    def initialize(self):
        with self.connection() as connection:
            connection.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(self.schema)))
            connection.execute(DDL)

    @contextmanager
    def snapshot_lock(self, *, publish: bool = False):
        # Readers hold shared session locks through vector search/answer fetch.
        # Publisher holds an exclusive session lock across DB commit + alias switch.
        with self.connection() as connection:
            operation = "pg_advisory_lock" if publish else "pg_advisory_lock_shared"
            release = "pg_advisory_unlock" if publish else "pg_advisory_unlock_shared"
            connection.execute(f"SELECT {operation}(%s)", (self.lock_id,))
            try:
                yield
            finally:
                if connection.info.transaction_status == psycopg.pq.TransactionStatus.INERROR:
                    connection.rollback()
                connection.execute(f"SELECT {release}(%s)", (self.lock_id,))

    def replace(self, dataset, metadata):
        with self.connection() as connection:
            connection.execute("DELETE FROM questions")
            connection.execute("DELETE FROM answers")
            with connection.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO answers VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    [
                        (a.id, a.category_id, a.topic_id, a.answer, a.status, a.created_at, a.updated_at)
                        for a in dataset.answers.values()
                    ],
                )
                cursor.executemany(
                    "INSERT INTO questions VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    [
                        (
                            q.id,
                            q.answer_id,
                            q.question,
                            q.embedding_text,
                            q.is_active,
                            q.created_at,
                            q.updated_at,
                        )
                        for q in dataset.questions.values()
                    ],
                )
            connection.execute(
                "INSERT INTO metadata VALUES (1,%s) ON CONFLICT(id) DO UPDATE SET value=EXCLUDED.value",
                (Jsonb(metadata),),
            )

    def metadata(self):
        with self.connection() as connection:
            row = connection.execute("SELECT value FROM metadata WHERE id=1").fetchone()
            if row is None:
                raise RuntimeError("No published dataset")
            return row["value"]

    def counts(self):
        with self.connection() as connection:
            return {
                name: connection.execute(f"SELECT count(*) AS n FROM {name}").fetchone()["n"]
                for name in ("questions", "answers")
            }

    def answer(self, ident: int):
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM answers WHERE id=%s", (ident,)).fetchone()
            if row is None or row["status"] != "active":
                raise RuntimeError("Accepted answer missing or inactive")
            return Answer(**row)

    def question(self, ident: int):
        with self.connection() as connection:
            row = connection.execute("SELECT question FROM questions WHERE id=%s", (ident,)).fetchone()
            if row is None:
                raise RuntimeError("Matched question missing")
            return row["question"]
