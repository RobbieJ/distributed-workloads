"""Feast-based RAG retriever for vector similarity search.

Ported from the kfto-sft-feast-rag example's feast_rag_retriever.py module.
Provides FeastRAGRetriever and FeastIndex classes for integrating Feast's
Milvus-backed vector store with HuggingFace's RAG models.
"""

import numpy as np
import torch
from transformers import RagRetriever


class FeastIndex:
    """Dummy index object required by RagRetriever interface."""

    def __init__(self):
        self.index_name = "feast_dummy_index"

    def get_doc_dicts(self, doc_ids):
        return [{"title": "", "text": ""} for _ in doc_ids]


class FeastRAGRetriever(RagRetriever):
    """RAG retriever backed by Feast feature store with Milvus vector search."""

    def __init__(
        self,
        question_encoder_tokenizer,
        generator_tokenizer,
        feast_repo_path,
        feature_view,
        features,
        search_type="vector",
        config=None,
        index=None,
        question_encoder=None,
        generator_model=None,
    ):
        super().__init__(
            config=config,
            question_encoder_tokenizer=question_encoder_tokenizer,
            generator_tokenizer=generator_tokenizer,
            index=index,
            init_retrieval=False,
        )
        from feast import FeatureStore

        self.store = FeatureStore(repo_path=feast_repo_path)
        self.feature_view = feature_view
        self.features = features
        self.search_type = search_type
        self.question_encoder = question_encoder
        self.generator_model = generator_model

    def retrieve(self, question_hidden_states, n_docs):
        """Retrieve documents using Feast vector similarity search."""
        query_vector = question_hidden_states[0].tolist()

        results = self.store.retrieve_online_documents_v2(
            features=[f"{self.feature_view.name}:embedding"],
            query=query_vector,
            top_k=n_docs,
        ).to_dict()

        doc_ids = []
        doc_texts = []
        doc_embeddings = []

        passage_ids = results.get("passage_id", [])
        passage_texts = results.get("passage_text", [])
        embeddings = results.get("embedding", [])

        for i in range(len(passage_ids)):
            doc_ids.append(i)  # Use integer IDs for RAG compatibility
            doc_texts.append(passage_texts[i] if i < len(passage_texts) else "")
            if i < len(embeddings):
                emb = embeddings[i]
                if isinstance(emb, list):
                    doc_embeddings.append(emb)
                else:
                    doc_embeddings.append([0.0] * 768)
            else:
                doc_embeddings.append([0.0] * 768)

        # Pad if fewer results than requested
        while len(doc_ids) < n_docs:
            doc_ids.append(len(doc_ids))
            doc_texts.append("")
            doc_embeddings.append([0.0] * 768)

        retrieved_doc_embeds = np.array(doc_embeddings[:n_docs], dtype=np.float32)
        # RAG model expects docs as [{"title": [list], "text": [list]}] per batch item
        doc_dicts = [{"title": [""] * n_docs, "text": doc_texts[:n_docs]}]

        return (
            np.array([retrieved_doc_embeds]),
            np.array([doc_ids[:n_docs]]),
            doc_dicts,
        )

    def is_question_answerable(
        self,
        question_text,
        expected_answer,
        question_encoder,
        tokenizer,
        top_k=10,
        similarity_threshold=0.6,
        max_answer_length=50,
        min_answer_length=1,
    ):
        """Check if a question can be answered using the knowledge base."""
        if not expected_answer or len(expected_answer.strip()) < min_answer_length:
            return False
        if len(expected_answer.strip()) > max_answer_length:
            return False

        inputs = tokenizer(question_text, return_tensors="pt", truncation=True, max_length=32)
        device = next(question_encoder.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            q_embedding = question_encoder(**inputs).pooler_output.cpu().numpy()

        _, _, doc_dicts = self.retrieve(q_embedding, top_k)

        answer_lower = expected_answer.lower().strip()
        for doc in doc_dicts:
            if answer_lower in doc.get("text", "").lower():
                return True
        return False
