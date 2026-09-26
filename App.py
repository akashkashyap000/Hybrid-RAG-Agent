from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chroma needs newer sqlite than Streamlit Cloud ships
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import html
import streamlit as st
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_community.tools import DuckDuckGoSearchRun

PERSIST_DIR = "chroma_db"
COLLECTION = None
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]

SYSTEM_PROMPT = (
    "You are a research assistant like Perplexity. Use search_documents for questions "
    "about the user's documents and web_search for current/general info. "
    "Each tool result is numbered like [1], [2]. Write a clear, well-structured answer "
    "using only those results and cite them inline like [1] or [2][3]. "
    "If the info is not found, say 'I don't have enough info.'"
)

SUGGESTIONS = [
    "Fundamental rights kya hain?",
    "SQL JOINs explain karo",
    "Overfitting kya hota hai?",
    "Power BI mein DAX kya hai?",
]

st.set_page_config(page_title="Ask", page_icon="🔍", layout="centered")

st.markdown(
    """
    <style>
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 760px; padding-top: 2.5rem; }
    .logo { text-align:center; font-size:44px; font-weight:300; letter-spacing:-1px; margin-top:12vh; }
    .logo b { color:#20b8cd; font-weight:600; }
    .tag { text-align:center; color:#8b8f8f; margin-bottom:26px; }
    .q { font-size:30px; font-weight:500; margin:8px 0 18px; line-height:1.25; }
    .lbl { color:#8b8f8f; font-size:13px; font-weight:600; letter-spacing:.5px;
      text-transform:uppercase; margin:6px 0 8px; }
    .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(170px,1fr));
      gap:10px; margin-bottom:18px; }
    .src { background:#202222; border:1px solid #2e3030; border-radius:12px; padding:10px 12px; }
    .src:hover { border-color:#20b8cd; }
    .src .t { font-size:12px; color:#cfd1d1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .src .s { font-size:11px; color:#8b8f8f; margin-top:4px; display:flex; align-items:center; gap:6px; }
    .n { background:#2e3030; color:#20b8cd; border-radius:50%; width:18px; height:18px;
      display:inline-flex; align-items:center; justify-content:center; font-size:10px; font-weight:700; }
    .tool { display:inline-block; font-size:11px; color:#20b8cd; border:1px solid #20b8cd55;
      border-radius:999px; padding:2px 10px; margin-right:6px; }
    hr.sep { border:none; border-top:1px solid #2e3030; margin:26px 0; }
    div[data-testid="stForm"] { border:1px solid #2e3030; border-radius:16px;
      background:#202222; padding:6px 10px; }
    div[data-testid="stForm"]:focus-within { border-color:#20b8cd; }
    div[data-testid="stForm"] input { font-size:17px; }
    </style>
    """,
    unsafe_allow_html=True,
)

api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

if not api_key:
    st.error("GROQ_API_KEY missing. Streamlit Cloud secrets mein GROQ_API_KEY daalo.")
    st.stop()

os.environ["GROQ_API_KEY"] = api_key


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_vectorstore():
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    kw = {"collection_name": COLLECTION} if COLLECTION else {}

    if os.path.exists(PERSIST_DIR):
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=emb,
            **kw
        )

    pdf_dir = "Data"
    all_docs = []

    if not os.path.exists(pdf_dir):
        raise RuntimeError("Data folder not found.")

    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(pdf_dir, filename)
            loader = PyPDFLoader(path)
            docs = loader.load()
            all_docs.extend(docs)

    if not all_docs:
        raise RuntimeError("No PDF files found inside the Data folder.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(all_docs)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=emb,
        persist_directory=PERSIST_DIR,
        **kw
    )

    return vectordb


vectordb = load_vectorstore()
web = DuckDuckGoSearchRun()


def run_agent(question, k, temperature, model):
    sources, tools_used = [], []
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    @tool
    def search_documents(query: str) -> str:
        """Search the user's ingested documents for relevant info."""
        docs = retriever.invoke(query)
        out = []

        for d in docs:
            n = len(sources) + 1
            pg = d.metadata.get("page")

            sources.append({
                "n": n,
                "type": "doc",
                "title": os.path.basename(
                    str(d.metadata.get("source", "unknown"))
                ),
                "meta": f"page {pg + 1}" if isinstance(pg, int) else "document",
                "text": d.page_content[:300],
            })

            out.append(f"[{n}] {d.page_content}")

        return "\n\n".join(out) or "No results."

    @tool
    def web_search(query: str) -> str:
        """Search the web for current or general information."""
        n = len(sources) + 1
        res = str(web.invoke(query))

        sources.append({
            "n": n,
            "type": "web",
            "title": f"Web: {query[:40]}",
            "meta": "web search",
            "text": res[:300],
        })

        return f"[{n}] {res}"

    tools = {
        "search_documents": search_documents,
        "web_search": web_search
    }

    llm = ChatGroq(
        model=model,
        temperature=temperature
    ).bind_tools(list(tools.values()))

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question)
    ]

    ai_msg = llm.invoke(messages)
    steps = 0

    while ai_msg.tool_calls and steps < 4:
        messages.append(ai_msg)

        for call in ai_msg.tool_calls:
            tools_used.append(call["name"])

            result = tools[call["name"]].invoke(call["args"])

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call["id"]
                )
            )

        ai_msg = llm.invoke(messages)
        steps += 1

    return ai_msg.content, sources, sorted(set(tools_used))


def render_turn(t):
    st.markdown(
        f'<div class="q">{html.escape(t["q"])}</div>',
        unsafe_allow_html=True
    )

    if t["sources"]:
        st.markdown(
            '<div class="lbl">Sources</div>',
            unsafe_allow_html=True
        )

        cards = "".join(
            f'<div class="src" title="{html.escape(s["text"])}">'
            f'<div class="t">{html.escape(s["title"])}</div>'
            f'<div class="s"><span class="n">{s["n"]}</span>'
            f'{html.escape(s["meta"])}</div></div>'
            for s in t["sources"]
        )

        st.markdown(
            f'<div class="grid">{cards}</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="lbl">Answer</div>',
        unsafe_allow_html=True
    )

    st.markdown(t["answer"])

    if t["tools"]:
        st.markdown(
            "".join(
                f'<span class="tool">{html.escape(x)}</span>'
                for x in t["tools"]
            ),
            unsafe_allow_html=True
        )

    st.markdown(
        '<hr class="sep">',
        unsafe_allow_html=True
    )


if "turns" not in st.session_state:
    st.session_state.turns = []

if "pending" not in st.session_state:
    st.session_state.pending = None


with st.sidebar:
    st.markdown("### 🔍 Ask")

    if st.button("＋ New thread", use_container_width=True):
        st.session_state.turns = []
        st.rerun()

    st.markdown("---")

    model = st.selectbox("Model", MODELS)
    k = st.slider("Sources to retrieve", 1, 10, 4)
    temperature = st.slider("Creativity", 0.0, 1.0, 0.0, 0.1)


def search_box(placeholder):
    with st.form("search", clear_on_submit=True):
        c1, c2 = st.columns([10, 1.4])

        text = c1.text_input(
            "q",
            placeholder=placeholder,
            label_visibility="collapsed"
        )

        go = c2.form_submit_button(
            "➜",
            use_container_width=True
        )

    return text.strip() if go and text.strip() else None


query = None

if not st.session_state.turns and not st.session_state.pending:
    st.markdown(
        '<div class="logo">know<b>ledge</b></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="tag">Ask anything from your documents or the web</div>',
        unsafe_allow_html=True
    )

    query = search_box("Ask anything...")

    cols = st.columns(2)

    for i, s in enumerate(SUGGESTIONS):
        if cols[i % 2].button(
            s,
            key=f"sg{i}",
            use_container_width=True
        ):
            st.session_state.pending = s
            st.rerun()

else:
    for t in st.session_state.turns:
        render_turn(t)

    query = search_box("Ask a follow-up...")


query = query or st.session_state.pending
st.session_state.pending = None

if query:
    with st.spinner("Searching your knowledge base..."):
        try:
            answer, sources, tools_used = run_agent(
                query,
                k,
                temperature,
                model
            )
        except Exception as e:
            answer = f"Error: {e}"
            sources = []
            tools_used = []

    st.session_state.turns.append({
        "q": query,
        "answer": answer,
        "sources": sources,
        "tools": tools_used
    })

    st.rerun()
