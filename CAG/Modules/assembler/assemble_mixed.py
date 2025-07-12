import numpy as np
import pandas as pd
from typing import List, Optional

class BaseAssembler:
    def assemble(self, chunks):
        raise NotImplementedError

    def merge_with_titles(self, structured_chunks, title_key="title", content_key="content"):
        merged = []
        for item in structured_chunks:
            title = item.get(title_key, "Untitled")
            content = item.get(content_key, "")
            merged.append(f"### {title}\n{content}")
        return "\n\n".join(merged)

class MixedAssembler(BaseAssembler):
    def __init__(self, priority: Optional[List[str]] = None):
        self.priority = priority or []

    def assemble(self, chunks):
        grouped = {}
        for c in chunks:
            dtype = self._detect_type(c)
            grouped.setdefault(dtype, []).append(c)

        ordered = []
        for p in self.priority:
            if p in grouped:
                ordered.extend(grouped[p])
        return ordered

    def _detect_type(self, chunk):
        if isinstance(chunk, str):
            return "text"
        elif isinstance(chunk, dict):
            return "table"
        elif "PIL" in str(type(chunk)):
            return "image"
        elif "AudioSegment" in str(type(chunk)):
            return "audio"
        elif isinstance(chunk, np.ndarray):
            return "image_array"
        else:
            return "unknown"


class ContextAssembler:
    def __init__(self, mode="text", sort_key=None, priority=None):
        self.mode = mode
        self.sort_key = sort_key
        self.priority = priority

        self.assemblers = {
            "mixed": MixedAssembler(priority=priority),
        }

        if mode not in self.assemblers:
            raise ValueError(f"Unsupported mode: {mode}")

    def assemble(self, chunks):
        if not chunks:
            return ""
        return self.assemblers[self.mode].assemble(chunks)

    def merge_with_titles(self, structured_chunks, title_key="title", content_key="content"):
        return self.assemblers[self.mode].merge_with_titles(structured_chunks, title_key, content_key)

    def build_prompt(self, chunks, query, instruction=None):
        assembled = self.assemble(chunks)
        prompt = ""

        if instruction:
            prompt += f"[Instruction]\n{instruction.strip()}\n\n"

        if isinstance(assembled, pd.DataFrame):
            prompt += f"[Context - Table]\n{assembled.to_string(index=False)}\n\n"
        elif isinstance(assembled, list):
            prompt += "[Context - List]\n" + "\n".join(str(x) for x in assembled) + "\n\n"
        else:
            prompt += f"[Context]\n{assembled.strip()}\n\n"

        prompt += f"[Question]\n{query.strip()}\n\n[Answer]"
        return prompt
