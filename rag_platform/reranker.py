"""Optional neural cross-encoder; externally downloaded weights, CPU inference."""


class CrossEncoderReranker:
    def __init__(self, model="cross-encoder/ms-marco-MiniLM-L6-v2"):
        from sentence_transformers import CrossEncoder

        self.name = model
        self.model = CrossEncoder(
            model,
            device="cpu",
            trust_remote_code=False,
            revision="233902d25c440f23af6f7d6e94d2946bac0bee0a"
            if model == "cross-encoder/ms-marco-MiniLM-L6-v2"
            else None,
        )

    def score(self, query, texts):
        return self.model.predict([(query, text) for text in texts], show_progress_bar=False)
