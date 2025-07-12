class BaseCompressionStrategy:
    def compress(self, chunks, query=None):
        raise NotImplementedError("Subclasses must implement compress method.")

class SummarizationCompressor(BaseCompressionStrategy):
    def __init__(self, model_name="facebook/bart-large-cnn", max_tokens=512):
        self.pipeline = pipeline("summarization", model=model_name)
        self.max_tokens = max_tokens

    def compress(self, chunks, query=None):
        text = "\n".join(chunks)
        result = self.pipeline(text, max_length=self.max_tokens, min_length=30, do_sample=False)
        return result[0]["summary_text"]
