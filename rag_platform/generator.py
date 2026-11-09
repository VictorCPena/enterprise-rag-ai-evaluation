"""Local seq2seq generator with an exact-source gate and extractive fallback."""

import time
from rag_platform.store import BAD


class LocalGenerator:
    def __init__(self, model="google/flan-t5-small", local_files_only=False):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        torch.set_num_threads(2)
        self.name = model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, revision="0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab", local_files_only=local_files_only
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model, revision="0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab", local_files_only=local_files_only
        )
        self.model.eval()

    def generate(self, question, hits):
        import torch

        if BAD.search(question) or not hits:
            return {"accepted": False, "reason": "unsafe query or no evidence"}
        context = "\n".join(h["text"][:700] for h in hits[:3] if not BAD.search(h["text"]))
        prompt = (
            "Answer using only an exact short quote from the context. If unsupported, answer UNKNOWN.\nQuestion: "
            + question
            + "\nContext: "
            + context
            + "\nAnswer:"
        )
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**encoded, max_new_tokens=60, do_sample=False)
        candidate = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        # The gate accepts only an exact span. This does not prove answer completeness.
        for h in hits:
            if len(candidate) >= 3 and candidate in h["text"] and not BAD.search(candidate):
                offset = h["text"].index(candidate)
                return {
                    "accepted": True,
                    "answer": candidate,
                    "citation": {
                        "doc": h["doc"],
                        "chunk": h["id"],
                        "url": h["url"],
                        "quote": candidate,
                        "source_sha256": h["sha"],
                        "start": h["start"] + offset,
                        "end": h["start"] + offset + len(candidate),
                    },
                    "seconds": time.perf_counter() - started,
                    "model": self.name,
                }
        return {
            "accepted": False,
            "reason": "Generated text was not an exact source span",
            "seconds": time.perf_counter() - started,
            "model": self.name,
        }


def grounded_answer(store, generator, question, k=5, strategy="hybrid", tenant="public"):
    hits = store.evidence_hits(question, k, strategy, tenant)
    draft = (
        generator.generate(question, hits)
        if hits
        else {"accepted": False, "reason": "No source span passed the evidence policy"}
    )
    if draft["accepted"]:
        return {
            "answer": draft["answer"] + " [1]",
            "citations": [{"id": 1, **draft["citation"]}],
            "abstained": False,
            "mode": "local-llm-exact-span",
            "generation": {k: v for k, v in draft.items() if k not in ["answer", "citation"]},
        }
    response = store.answer(question, k, strategy, tenant)
    response["generation"] = draft
    response["mode"] = "extractive-fallback"
    return response
