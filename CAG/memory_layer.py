
from typing import List, Dict, Optional, Literal
import redis
import json

class MemoryLayer:
    """
    Manages conversation history using a Redis database for persistent,
    scalable memory. It correctly handles roles for structured conversations.
    """
    def __init__(self, session_id: str, host: str = 'localhost', port: int = 6379, db: int = 0, max_len: int = 20):
        self.session_id = f"history:{session_id}"
        self.max_len = max_len
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)

    def add_message(self, role: Literal["user", "assistant", "system"], content: str) -> None:
        """
        Adds a new message with a specific role to the history in Redis.
        The 'system' role can be used to set the AI's persona.
        """
        message = json.dumps({"role": role, "content": content})
        self.client.rpush(self.session_id, message)
        self.client.ltrim(self.session_id, -self.max_len, -1)

    def get_history(self) -> List[Dict[str, str]]:
        """Retrieves the entire conversation history for the session from Redis."""
        history_json = self.client.lrange(self.session_id, 0, -1)
        return [json.loads(msg) for msg in history_json]

    def format_for_prompt(self) -> List[str]:
        """
        Formats the history into a human-readable list of strings,
        perfect for passing to the ContextCompressor.
        """
        history = self.get_history()
        return [f"{msg['role'].capitalize()}: {msg['content']}" for msg in history]

    def modern_format(self) -> List[Dict[str, str]]:
        """
        Formats the history into the dictionary list format expected
        by modern LLMs.
        """
        return self.get_history()

    def clear(self) -> None:
        """Deletes the entire conversation history for this session from Redis."""
        self.client.delete(self.session_id)