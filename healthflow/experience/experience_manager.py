import sqlite3
from pathlib import Path
from typing import List
from loguru import logger

from .experience_models import Experience

class ExperienceManager:
    """
    Manages the persistent storage and retrieval of structured experiences
    in a SQLite database. This forms the system's long-term memory.
    """
    def __init__(self, workspace_dir: str):
        db_path = Path(workspace_dir) / "experience.db"
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_table()
        logger.info(f"ExperienceManager initialized. Database at: {db_path}")

    def _create_table(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_task_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # For future FTS (Full-Text Search)
            self.conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS experiences_fts
                USING fts5(content, tokenize = 'porter unicode61');
            """)

    async def save_experiences(self, experiences: List[Experience]):
        """Saves a list of new experiences to the database."""
        if not experiences:
            return

        values = [(exp.type.value, exp.category, exp.content, exp.source_task_id) for exp in experiences]

        with self.conn:
            cursor = self.conn.cursor()
            cursor.executemany(
                "INSERT INTO experiences (type, category, content, source_task_id) VALUES (?, ?, ?, ?)",
                values
            )
            # Add to FTS table
            for exp in experiences:
                cursor.execute(
                    "INSERT INTO experiences_fts (rowid, content) VALUES (?, ?)",
                    (cursor.lastrowid, exp.content)
                )
        logger.info(f"Saved {len(experiences)} new experiences to the database.")

    async def retrieve_experiences(self, query: str, k: int = 5) -> List[Experience]:
        """
        Retrieves the top k most relevant experiences using simple keyword matching.
        NOTE: This is a placeholder for a more advanced vector search / RAG implementation.
        """
        # Simple FTS5-based search
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT e.* FROM experiences e
                JOIN experiences_fts fts ON e.id = fts.rowid
                WHERE fts.content MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (query, k))

            rows = cursor.fetchall()

        experiences = [Experience.from_db_row(row) for row in rows]
        return experiences

    def __del__(self):
        if self.conn:
            self.conn.close()