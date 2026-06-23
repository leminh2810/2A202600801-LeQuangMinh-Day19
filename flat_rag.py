import os
import math
import re
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY", "")
is_mock_llm = not api_key or api_key.startswith("your_openai_api")

# Validate key with a test call if api_key is present
client = None
if not is_mock_llm:
    try:
        temp_client = OpenAI(api_key=api_key)
        # Try a quick test call to verify key
        temp_client.embeddings.create(input=["test"], model="text-embedding-ada-002")
        client = temp_client
        print("OpenAI API key validated successfully.")
    except Exception as e:
        print(f"OpenAI API key validation failed: {e}. Falling back to MOCK MODE.")
        is_mock_llm = True
        client = None

def load_corpus(file_path="corpus.txt"):
    """Reads paragraphs from corpus.txt."""
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    return paragraphs

# Lightweight TF-IDF implementation for mock fallback
def tokenize(text):
    text = text.lower()
    # Find all words
    words = re.findall(r"\w+", text)
    return words

class SimpleTFIDFRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.doc_tokens = [tokenize(doc) for doc in documents]
        self.vocab = set(word for tokens in self.doc_tokens for word in tokens)
        self.vocab = list(self.vocab)
        self.vocab_index = {word: i for i, word in enumerate(self.vocab)}
        
        # Calculate IDF
        self.idf = {}
        n_docs = len(documents)
        for word in self.vocab:
            doc_count = sum(1 for tokens in self.doc_tokens if word in tokens)
            self.idf[word] = math.log((1 + n_docs) / (1 + doc_count)) + 1
            
        # Calculate document TF-IDF vectors
        self.doc_vectors = []
        for tokens in self.doc_tokens:
            vec = self.get_tfidf_vector(tokens)
            self.doc_vectors.append(vec)
            
    def get_tfidf_vector(self, tokens):
        vector = [0.0] * len(self.vocab)
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        for token, count in token_counts.items():
            if token in self.vocab_index:
                tf = count / len(tokens)
                idx = self.vocab_index[token]
                vector[idx] = tf * self.idf[token]
        return vector

    def cosine_similarity(self, vec1, vec2):
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = math.sqrt(sum(a * a for a in vec1))
        norm_b = math.sqrt(sum(b * b for b in vec2))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def retrieve(self, query, top_k=2):
        query_tokens = tokenize(query)
        query_vec = self.get_tfidf_vector(query_tokens)
        
        scores = []
        for i, doc_vec in enumerate(self.doc_vectors):
            sim = self.cosine_similarity(query_vec, doc_vec)
            scores.append((self.documents[i], sim))
            
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scores[:top_k]]

# Custom Embedding Function class to satisfy ChromaDB signature check
class OpenAIEmbeddingFunction(chromadb.EmbeddingFunction):
    def __init__(self, client):
        self.client = client
        
    def __call__(self, input):
        embeddings = []
        for text in input:
            try:
                response = self.client.embeddings.create(
                    input=[text],
                    model="text-embedding-ada-002"
                )
                embeddings.append(response.data[0].embedding)
            except Exception as e:
                print(f"Error creating embedding: {e}")
                # Return dummy zeros
                embeddings.append([0.0] * 1536)
        return embeddings

# ChromaDB handler for production (valid OpenAI Key)
class ChromaRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.chroma_client = chromadb.Client()
        # Create unique collection
        self.collection = self.chroma_client.create_collection(
            name="tech_company_corpus",
            embedding_function=OpenAIEmbeddingFunction(client)
        )
        # Populate DB
        ids = [f"doc_{i}" for i in range(len(documents))]
        self.collection.add(
            documents=documents,
            ids=ids
        )

    def retrieve(self, query, top_k=2):
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        return results["documents"][0]

# Global cache for retriever
retriever_instance = None

def init_retriever():
    global retriever_instance
    if retriever_instance is None:
        docs = load_corpus()
        if is_mock_llm:
            retriever_instance = SimpleTFIDFRetriever(docs)
        else:
            retriever_instance = ChromaRetriever(docs)

def query_flat_rag(query, top_k=2):
    """Performs standard Flat RAG querying using text chunks retrieval."""
    init_retriever()
    retrieved_chunks = retriever_instance.retrieve(query, top_k=top_k)
    
    context = "\n\n".join(retrieved_chunks)
    
    prompt = f"""
    Bạn là một trợ lý thông minh trả lời câu hỏi dựa trên các đoạn văn bản được cung cấp (Flat RAG Context).
    Hãy trả lời câu hỏi một cách ngắn gọn, chính xác bằng tiếng Việt dựa TRỰC TIẾP và CHỈ dựa trên ngữ cảnh được cung cấp.

    Ngữ cảnh:
    {context}

    Câu hỏi: "{query}"
    Câu trả lời:
    """
    
    if is_mock_llm:
        # Generate mock answer by returning the top retrieved chunk
        if not retrieved_chunks:
            return "Xin lỗi, tôi không tìm thấy thông tin phù hợp."
        return f"[Flat RAG Mock Output] Dựa trên văn bản thô đã tìm thấy:\n{context}\n\n-> Kết luận: Trích xuất nội dung văn bản thô liên quan đến câu hỏi của bạn."

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful QA assistant that answers queries strictly using the provided context paragraphs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Lỗi truy vấn Flat RAG: {e}"

if __name__ == "__main__":
    init_retriever()
    q = "Ai sáng lập OpenAI và mối liên hệ của họ với Microsoft?"
    ans = query_flat_rag(q)
    print("Query:", q)
    print("Answer:\n", ans)
