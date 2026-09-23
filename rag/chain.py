from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from rag.vectorstore import get_vectorstore
from llm.model import get_openrouter_model

TOP_K = 3
MIN_RELEVANCE_SCORE = 0.45
DIRECT_ANSWER_SCORE = 0.72

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You answer questions for the Barangay Bagumbayan Health Center.

Use only the provided context. If the context does not contain enough
information to answer, say: "I don't have that information."

Keep the answer concise, factual, and safe. Do not make up schedules,
contact details, medical diagnoses, treatments, fees, or policies.
""".strip(),
        ),
        (
            "human",
            """
Question:
{question}

Context:
{context}
""".strip(),
        ),
    ]
)


class RAGChain:
    def __init__(
        self,
        vectorstore: FAISS,
        top_k: int = TOP_K,
        min_relevance_score: float = MIN_RELEVANCE_SCORE,
    ):
        self.vectorstore = vectorstore
        self.llm = get_openrouter_model()
        self.top_k = top_k
        self.min_relevance_score = min_relevance_score
        self._cache = {}

    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        question = inputs.get("query") or inputs.get("question")

        if not question:
            raise ValueError("RAGChain requires a `query` or `question` input.")

        cache_key = self._cache_key(question)
        if cache_key in self._cache:
            return self._cache[cache_key]

        source_documents = self._retrieve(question)

        if not source_documents:
            result = {
                "result": "I don't have that information.",
                "source_documents": [],
            }
            self._cache[cache_key] = result
            return result

        direct_answer = self._direct_answer(source_documents[0])
        if direct_answer:
            result = {
                "result": direct_answer,
                "source_documents": [source_documents[0]],
            }
            self._cache[cache_key] = result
            return result

        context = self._format_context(source_documents)
        messages = RAG_PROMPT.format_messages(question=question, context=context)
        response = self.llm.invoke(messages)

        result = {
            "result": response.content,
            "source_documents": source_documents,
        }
        self._cache[cache_key] = result
        return result

    def _retrieve(self, question: str) -> list[Document]:
        try:
            results = self.vectorstore.similarity_search_with_relevance_scores(
                question,
                k=self.top_k,
            )

            return [
                self._with_score(doc, score)
                for doc, score in results
                if score >= self.min_relevance_score
            ]
        except Exception:
            return self.vectorstore.similarity_search(question, k=self.top_k)

    def _with_score(self, document: Document, score: float) -> Document:
        document.metadata = {
            **document.metadata,
            "relevance_score": round(score, 4),
        }
        return document

    def _direct_answer(self, document: Document) -> str | None:
        score = document.metadata.get("relevance_score")
        answer = document.metadata.get("answer")

        if score is not None and score >= DIRECT_ANSWER_SCORE and answer:
            return answer

        return None

    def _format_context(self, documents: list[Document]) -> str:
        return "\n\n".join(
            f"Source {index}\n{document.page_content}"
            for index, document in enumerate(documents, start=1)
        )

    def _cache_key(self, question: str) -> str:
        return " ".join(question.lower().strip().split())


_vectorstore = None
_qa_chain = None


def create_rag_chain(vectorstore: FAISS):
    return RAGChain(vectorstore)


def get_qa_chain():
    global _vectorstore, _qa_chain

    if _qa_chain is None:
        _vectorstore = get_vectorstore()
        _qa_chain = create_rag_chain(_vectorstore)

    return _qa_chain
