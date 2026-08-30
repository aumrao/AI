import os
import re
import time
import threading
from typing import List, Optional, Dict


class GeminiKeyPool:
    """
    Manages a pool of multiple Gemini API keys.
    Provides thread-safe round-robin allocation, rate-limit cooldown tracking, and parallel key distribution.
    """
    def __init__(self, keys: Optional[List[str]] = None):
        self._lock = threading.Lock()
        self._index = 0
        self._cooldowns: Dict[str, float] = {}  # key -> cooldown_expiry_timestamp
        self.keys = []
        if keys:
            self.set_keys(keys)
        else:
            env_keys = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY", "")
            self.set_keys(self.parse_keys_str(env_keys))

    @staticmethod
    def parse_keys_str(keys_str: str) -> List[str]:
        """Parses comma, newline, or space-separated API keys into a clean unique list."""
        if not keys_str:
            return []
        raw = re.split(r'[\s,;\n\r]+', keys_str.strip())
        # Filter out empty or whitespace tokens
        cleaned = [k.strip() for k in raw if k.strip() and len(k.strip()) > 10]
        # Deduplicate while preserving order
        return list(dict.fromkeys(cleaned))

    def set_keys(self, keys: List[str]):
        with self._lock:
            self.keys = [k.strip() for k in keys if k and k.strip()]
            self._index = 0

    def add_key(self, key: str):
        with self._lock:
            k = key.strip()
            if k and k not in self.keys:
                self.keys.append(k)

    def size(self) -> int:
        return len(self.keys)

    def get_all_keys(self) -> List[str]:
        return list(self.keys)

    def get_available_keys(self) -> List[str]:
        """Returns all keys currently not in rate-limit cooldown."""
        now = time.time()
        with self._lock:
            return [k for k in self.keys if self._cooldowns.get(k, 0) <= now]

    def get_next_key(self) -> Optional[str]:
        """Returns next available key using round-robin rotation."""
        with self._lock:
            if not self.keys:
                return None

            now = time.time()
            # Try to find a non-cooling-down key
            available = [k for k in self.keys if self._cooldowns.get(k, 0) <= now]
            if not available:
                # If all are cooling down, use the least recently rate-limited key
                available = self.keys

            self._index = (self._index + 1) % len(available)
            return available[self._index]

    def mark_rate_limited(self, key: str, cooldown_seconds: float = 60.0):
        """Marks a key as rate-limited for cooldown_seconds."""
        with self._lock:
            self._cooldowns[key] = time.time() + cooldown_seconds
            print(f"Notice: Gemini API Key ({key[:6]}...{key[-4:]}) cooling down for {cooldown_seconds}s.")

    def get_sharded_keys(self, num_shards: int) -> List[str]:
        """
        Returns a list of keys of length num_shards.
        If num_shards > pool size, keys are recycled/round-robined.
        """
        if not self.keys:
            return []
        keys = self.get_available_keys() or self.keys
        return [keys[i % len(keys)] for i in range(num_shards)]
