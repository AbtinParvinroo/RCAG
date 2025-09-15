from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import logging
try:
    from PIL.Image import Image

except ImportError:
    Image = None

try:
    from pydub import AudioSegment

except ImportError:
    AudioSegment = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ContextAssembler:
    """
    Organizes a list of heterogeneous chunks into a structured dictionary based on their data type.
    This class is responsible for the "what" goes into the context, not the "how" it's formatted.
    """
    def __init__(self, priority: Optional[List[str]] = None):
        """
        Initializes the assembler.
        Args:
            priority: An optional list of types (e.g., ["text", "image"]) to determine the order
            of chunks in the final assembled list.
        """
        self.priority = priority or []
        logger.info(f"ContextAssembler initialized with priority: {self.priority}")

    def _detect_type(self, chunk: Any) -> str:
        """Robustly detects the data type of a single chunk."""
        if isinstance(chunk, str):
            return "text"

        if Image and isinstance(chunk, Image):
            return "image"

        if AudioSegment and isinstance(chunk, AudioSegment):
            return "audio"

        if isinstance(chunk, pd.DataFrame):
            return "table"

        if isinstance(chunk, dict):
            return "structured_data" # e.g., a row from a table

        if isinstance(chunk, np.ndarray):
            return "image_array"

        return "unknown"

    def assemble(self, chunks: List[Any]) -> Dict[str, List[Any]]:
        """
        Assembles a list of chunks into a dictionary grouped by data type.
        Args:
            chunks: A list of data chunks of various types.
        Returns:
            A dictionary where keys are data types (e.g., 'text', 'image') and
            values are lists of chunks of that type.
        """
        if not chunks:
            return {}

        grouped: Dict[str, List[Any]] = {}
        for chunk in chunks:
            dtype = self._detect_type(chunk)
            if dtype != "unknown":
                grouped.setdefault(dtype, []).append(chunk)

        if self.priority:
            ordered_chunks = []
            for dtype in self.priority:
                if dtype in grouped:
                    ordered_chunks.extend(grouped[dtype])

            for dtype, data in grouped.items():
                if dtype not in self.priority:
                    ordered_chunks.extend(data)

            return {"mixed": ordered_chunks}

        return grouped

class PromptFormatter:
    """
    Takes a structured dictionary of context from ContextAssembler and builds a
    final, formatted string prompt ready for an LLM.
    """
    def build_prompt(self, assembled_context: Dict[str, List[Any]], query: str, instruction: Optional[str] = None) -> str:
        """
        Builds a complete prompt string from assembled context, a query, and an instruction.
        Args:
            assembled_context: The output from ContextAssembler.
            query: The user's question.
            instruction: An optional system prompt or instruction for the LLM.
        Returns:
            A single, formatted string ready to be sent to the LLM.
        """
        prompt_parts = []
        if instruction:
            prompt_parts.append(f"### Instruction\n{instruction.strip()}")

        context_str = self.format_context_to_string(assembled_context)
        if context_str:
            prompt_parts.append(f"### Context\n{context_str.strip()}")

        prompt_parts.append(f"### Question\n{query.strip()}")
        prompt_parts.append("### Answer")
        return "\n\n".join(prompt_parts)

    def format_context_to_string(self, assembled_context: Dict[str, List[Any]]) -> str:
        """Converts the assembled context dictionary into a formatted string."""
        formatted_parts = []
        for dtype, data in assembled_context.items():
            if not data:
                continue

            if dtype == "text":
                formatted_parts.append("\n".join(data))

            elif dtype == "table":
                for i, df in enumerate(data):
                    formatted_parts.append(f"--- Table {i+1} ---\n{df.to_string(index=False)}")
            elif dtype == "structured_data":
                df = pd.DataFrame(data)
                formatted_parts.append(f"--- Data ---\n{df.to_string(index=False)}")
            else:
                formatted_parts.append(f"--- {dtype.capitalize()} Content ---\n[{len(data)} {dtype} item(s) provided]")

        return "\n\n".join(formatted_parts)