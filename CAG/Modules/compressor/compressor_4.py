class BaseCompressionStrategy:
    def compress(self, chunks, query=None):
        raise NotImplementedError("Subclasses must implement compress method.")

class MergeAndTrimCompressor(BaseCompressionStrategy):
    def __init__(self, tokenizer_model="bert-base-uncased", max_tokens=512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.max_tokens = max_tokens

    def compress(self, chunks, query=None):
        text = " ".join(chunks)
        tokens = self.tokenizer.encode(text, truncation=True, max_length=self.max_tokens)
        return self.tokenizer.decode(tokens)

class ContextCompressor:
    def __init__(self, strategy: BaseCompressionStrategy):
        self.strategy = strategy

    def compress(self, chunks, query=None):
        return self.strategy.compress(chunks, query=query)
