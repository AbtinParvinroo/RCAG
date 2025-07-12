import pandas as pd

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

class ImageAssembler(BaseAssembler):
    def __init__(self, sort_key=None):
        self.sort_key = sort_key

    def assemble(self, chunks):
        return sorted(chunks, key=self.sort_key) if self.sort_key else chunks

class ContextAssembler:
    def __init__(self, mode="text", sort_key=None, priority=None):
        self.mode = mode
        self.sort_key = sort_key
        self.priority = priority

        self.assemblers = {
            "image": ImageAssembler(sort_key=sort_key),
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
