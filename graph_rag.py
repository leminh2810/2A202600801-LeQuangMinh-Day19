import os
import json
import re
import matplotlib.pyplot as plt
import networkx as nx
from dotenv import load_dotenv
from openai import OpenAI

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

# Pre-defined triples for mock fallback
MOCK_TRIPLES = [
    # Google
    ("Google", "FOUNDED_BY", "Larry Page"),
    ("Google", "FOUNDED_BY", "Sergey Brin"),
    ("Google", "FOUNDED_IN", "1998"),
    ("Google", "PARENT_COMPANY", "Alphabet Inc."),
    ("Alphabet Inc.", "CEO", "Sundar Pichai"),
    ("Google", "CEO", "Sundar Pichai"),
    ("Google", "ACQUIRED", "Android"),
    ("Android", "ACQUIRED_IN", "2005"),
    ("Google", "ACQUIRED", "YouTube"),
    ("YouTube", "ACQUIRED_IN", "2006"),
    
    # Apple
    ("Apple Inc.", "FOUNDED_BY", "Steve Jobs"),
    ("Apple Inc.", "FOUNDED_BY", "Steve Wozniak"),
    ("Apple Inc.", "FOUNDED_BY", "Ronald Wayne"),
    ("Apple Inc.", "FOUNDED_IN", "1976"),
    ("Apple Inc.", "LAUNCHED", "iPhone"),
    ("iPhone", "LAUNCHED_IN", "2007"),
    ("Apple Inc.", "CEO", "Steve Jobs"),
    ("Apple Inc.", "CEO", "Tim Cook"),
    ("Apple Inc.", "ACQUIRED", "Beats Electronics"),
    ("Beats Electronics", "ACQUIRED_IN", "2014"),

    # Microsoft
    ("Microsoft", "FOUNDED_BY", "Bill Gates"),
    ("Microsoft", "FOUNDED_BY", "Paul Allen"),
    ("Microsoft", "FOUNDED_IN", "1975"),
    ("Microsoft", "CEO", "Satya Nadella"),
    ("Microsoft", "CEO", "Steve Ballmer"),
    ("Microsoft", "ACQUIRED", "LinkedIn"),
    ("LinkedIn", "ACQUIRED_IN", "2016"),
    ("Microsoft", "ACQUIRED", "GitHub"),
    ("GitHub", "ACQUIRED_IN", "2018"),
    ("Microsoft", "ACQUIRED", "Activision Blizzard"),
    ("Activision Blizzard", "ACQUIRED_IN", "2023"),
    ("Microsoft", "INVESTED_IN", "OpenAI"),

    # OpenAI
    ("OpenAI", "FOUNDED_BY", "Sam Altman"),
    ("OpenAI", "FOUNDED_BY", "Elon Musk"),
    ("OpenAI", "FOUNDED_BY", "Ilya Sutskever"),
    ("OpenAI", "FOUNDED_BY", "Greg Brockman"),
    ("OpenAI", "FOUNDED_BY", "Wojciech Zaremba"),
    ("OpenAI", "FOUNDED_BY", "John Schulman"),
    ("OpenAI", "FOUNDED_IN", "2015"),
    ("OpenAI", "CEO", "Sam Altman"),
    ("Elon Musk", "LEFT_BOARD", "OpenAI"),
    ("OpenAI", "LAUNCHED", "ChatGPT"),
    ("ChatGPT", "LAUNCHED_IN", "2022"),

    # Meta
    ("Meta", "FOUNDED_BY", "Mark Zuckerberg"),
    ("Meta", "FOUNDED_IN", "2004"),
    ("Meta", "CEO", "Mark Zuckerberg"),
    ("Meta", "ACQUIRED", "Instagram"),
    ("Instagram", "ACQUIRED_IN", "2012"),
    ("Meta", "ACQUIRED", "WhatsApp"),
    ("WhatsApp", "ACQUIRED_IN", "2014"),
    ("Meta", "ACQUIRED", "Oculus VR"),
    ("Oculus VR", "ACQUIRED_IN", "2014"),

    # Nvidia
    ("Nvidia", "FOUNDED_BY", "Jensen Huang"),
    ("Nvidia", "FOUNDED_BY", "Chris Malachowsky"),
    ("Nvidia", "FOUNDED_BY", "Curtis Priem"),
    ("Nvidia", "FOUNDED_IN", "1993"),
    ("Nvidia", "CEO", "Jensen Huang"),
    ("Nvidia", "LAUNCHED", "CUDA"),
    ("CUDA", "LAUNCHED_IN", "2006"),
    ("Nvidia", "ACQUIRED", "Mellanox Technologies"),
    ("Mellanox Technologies", "ACQUIRED_IN", "2020")
]

def load_corpus(file_path="corpus.txt"):
    """Reads paragraphs from corpus.txt."""
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    # Split by empty lines to get paragraphs
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    return paragraphs

def extract_triples_with_llm(paragraph):
    """Extracts entity-relation triples from a paragraph using OpenAI LLM."""
    if is_mock_llm:
        # Filter mock triples that match entities in the paragraph
        # This is a fallback to make code run without api keys
        paragraph_lower = paragraph.lower()
        extracted = []
        for s, r, o in MOCK_TRIPLES:
            if s.lower()[:5] in paragraph_lower or o.lower()[:5] in paragraph_lower:
                extracted.append([s, r, o])
        return extracted

    prompt = f"""
    Bạn là một chuyên gia trích xuất dữ liệu đồ thị tri thức. 
    Từ đoạn văn bản sau đây, hãy trích xuất toàn bộ các thực thể (Entity) và quan hệ (Relation) giữa chúng dưới dạng bộ ba (Triples): (Thực thể 1, QUAN_HỆ, Thực thể 2).
    Lưu ý:
    1. Tên thực thể cần được chuẩn hóa (ví dụ: "Google Inc." -> "Google", "Facebook" -> "Meta").
    2. Tên quan hệ viết bằng chữ in hoa, phân tách bằng dấu gạch dưới (ví dụ: FOUNDED_BY, ACQUIRED, CEO, LAUNCHED).
    3. Trả về KẾT QUẢ duy nhất dưới dạng một JSON Array chứa các Array con dạng ["Thực thể 1", "QUAN_HỆ", "Thực thể 2"]. Không thêm bất kỳ văn bản giải thích nào khác.

    Văn bản: "{paragraph}"
    JSON Output:
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful data extraction assistant. You only output valid JSON arrays."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        result_text = response.choices[0].message.content.strip()
        # Find JSON array in text
        match = re.search(r"\[\s*\[.*\]\s*\]", result_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(result_text)
    except Exception as e:
        print(f"Error calling LLM for extraction, falling back to mock: {e}")
        # Fallback to mock filtering
        paragraph_lower = paragraph.lower()
        extracted = []
        for s, r, o in MOCK_TRIPLES:
            if s.lower()[:5] in paragraph_lower or o.lower()[:5] in paragraph_lower:
                extracted.append([s, r, o])
        return extracted

def build_knowledge_graph(paragraphs):
    """Builds a NetworkX directed graph from paragraphs."""
    G = nx.DiGraph()
    all_triples = []
    
    for p in paragraphs:
        triples = extract_triples_with_llm(p)
        for triple in triples:
            if len(triple) == 3:
                s, r, o = triple
                # Normalize node names
                s = s.strip()
                o = o.strip()
                r = r.strip().upper()
                all_triples.append((s, r, o))
                G.add_edge(s, o, relation=r)
                
    print(f"Đã xây dựng đồ thị với {G.number_of_nodes()} thực thể và {G.number_of_edges()} quan hệ.")
    return G, all_triples

def draw_graph(G, output_path="knowledge_graph.png"):
    """Visualizes the graph and saves it as an image."""
    plt.figure(figsize=(10, 8))
    
    # Position nodes using spring layout
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=1200, node_color="#4361ee", alpha=0.9)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=12, edge_color="#9381ff", width=1.2)
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight="bold", font_color="white")
    
    # Draw edge labels (relations)
    edge_labels = nx.get_edge_attributes(G, "relation")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, font_color="#3f37c9")
    
    plt.title("Tech Company Knowledge Graph", fontsize=12, fontweight="bold", pad=15)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Đã lưu hình ảnh đồ thị tại {output_path}")

def extract_query_entities(query):
    """Extracts entities from the user's query."""
    if is_mock_llm:
        # Simple keyword matching for mock fallback
        entities = []
        known_entities = ["google", "apple", "microsoft", "openai", "meta", "nvidia", 
                          "sam altman", "elon musk", "sundar pichai", "tim cook", 
                          "satya nadella", "mark zuckerberg", "jensen huang", "chatgpt"]
        query_lower = query.lower()
        for ent in known_entities:
            if ent in query_lower:
                # Find matching capitalized entity in mock database
                for s, _, o in MOCK_TRIPLES:
                    if s.lower() == ent:
                        entities.append(s)
                    if o.lower() == ent:
                        entities.append(o)
        return list(set(entities))

    prompt = f"""
    Bạn là một trợ lý phân tích câu hỏi. Hãy xác định các thực thể chính (như tên công ty, người sáng lập, sản phẩm, công nghệ) được nhắc đến trong câu hỏi dưới đây.
    Trả về kết quả dưới dạng một JSON Array chứa danh sách các thực thể được trích xuất. Không thêm giải thích gì thêm.

    Câu hỏi: "{query}"
    JSON Output:
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful entities extraction assistant. You only output valid JSON arrays."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        result_text = response.choices[0].message.content.strip()
        match = re.search(r"\[.*\]", result_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(result_text)
    except Exception as e:
        print(f"Error extracting query entities: {e}")
        return []

def get_2_hop_context(G, entities):
    """Retrieves 2-hop neighborhood facts for the given entities."""
    context_facts = []
    visited_edges = set()
    
    for entity in entities:
        if entity not in G:
            # Try partial matching if exact name is not in graph
            matched_nodes = [node for node in G if entity.lower() in node.lower() or node.lower() in entity.lower()]
            if not matched_nodes:
                continue
            entity = matched_nodes[0]
            
        # 1-hop outbound and inbound
        neighbors_1 = list(G.successors(entity)) + list(G.predecessors(entity))
        
        for n in neighbors_1:
            # Outbound edge
            if G.has_edge(entity, n):
                edge = (entity, n)
                if edge not in visited_edges:
                    visited_edges.add(edge)
                    rel = G[entity][n]["relation"]
                    context_facts.append(f"- ({entity}) có quan hệ [{rel}] tới ({n})")
            # Inbound edge
            if G.has_edge(n, entity):
                edge = (n, entity)
                if edge not in visited_edges:
                    visited_edges.add(edge)
                    rel = G[n][entity]["relation"]
                    context_facts.append(f"- ({n}) có quan hệ [{rel}] tới ({entity})")
                    
            # 2-hop outbound and inbound from the neighbor
            neighbors_2 = list(G.successors(n)) + list(G.predecessors(n))
            for n2 in neighbors_2:
                if G.has_edge(n, n2):
                    edge = (n, n2)
                    if edge not in visited_edges:
                        visited_edges.add(edge)
                        rel = G[n][n2]["relation"]
                        context_facts.append(f"- ({n}) có quan hệ [{rel}] tới ({n2})")
                if G.has_edge(n2, n):
                    edge = (n2, n)
                    if edge not in visited_edges:
                        visited_edges.add(edge)
                        rel = G[n2][n]["relation"]
                        context_facts.append(f"- ({n2}) có quan hệ [{rel}] tới ({n})")
                        
    return "\n".join(context_facts)

def query_graph_rag(G, query):
    """Performs GraphRAG querying by retrieving graph context and invoking the LLM."""
    entities = extract_query_entities(query)
    graph_context = get_2_hop_context(G, entities)
    
    if not graph_context:
        graph_context = "Không tìm thấy dữ liệu liên quan trong đồ thị tri thức."
        
    prompt = f"""
    Bạn là một trợ lý thông minh trả lời câu hỏi dựa trên Ngữ cảnh Đồ thị Tri thức (Knowledge Graph Context) dưới dạng các mối liên kết (Triples).
    Hãy trả lời câu hỏi một cách ngắn gọn, chính xác bằng tiếng Việt dựa TRỰC TIẾP và CHỈ dựa trên các mối liên kết được cung cấp.

    Ngữ cảnh Đồ thị Tri thức:
    {graph_context}

    Câu hỏi: "{query}"
    Câu trả lời:
    """
    
    if is_mock_llm:
        # Simple mock reasoning to generate a response based on the context facts
        # We parse the graph context and construct a descriptive text
        if "Không tìm thấy" in graph_context:
            return "Xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn."
            
        # We can construct a mock answer using a simple LLM-like template
        # Let's clean the facts and build a paragraph
        facts_clean = graph_context.replace("- (", "").replace(") có quan hệ [", " ").replace("] tới (", " ").replace(")", "")
        return f"[GraphRAG Mock Output] Dựa trên đồ thị tri thức:\n{graph_context}\n\n-> Kết luận: Đây là thông tin liên kết được tìm thấy trong đồ thị liên quan đến câu hỏi của bạn."

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful QA assistant that answers queries strictly using the provided graph context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Lỗi truy vấn GraphRAG: {e}"

if __name__ == "__main__":
    paragraphs = load_corpus()
    G, triples = build_knowledge_graph(paragraphs)
    draw_graph(G)
    
    # Test query
    q = "Ai sáng lập OpenAI và mối liên hệ của họ với Microsoft?"
    ans = query_graph_rag(G, q)
    print("\nQuery:", q)
    print("Answer:\n", ans)
