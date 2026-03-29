import os
import zipfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import YoutubeLoader
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ─────────────────────────────────────────────
# 1.  Model  (Gemini)
# ─────────────────────────────────────────────
def get_llm(api_key: str):
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.7,
    )


# ─────────────────────────────────────────────
# 2.  Article-writer prompts
# ─────────────────────────────────────────────
ARTICLE_SYSTEM = "You are a Professional Article Writer specialising in articles for Medium, LinkedIn, and tech blogs."

ARTICLE_HUMAN = """
Transform the YouTube transcript below into an **engaging, professional article**.

**CRITICAL INSTRUCTIONS**:
- **IGNORE** introductory notes like "welcome", "in this video"
- **IGNORE** channel names, "subscribe", "like", "comment", "follow", "check description"
- **IGNORE** marketing phrases: "my course", "my discord", "affiliate links", "sponsors"
- **FOCUS ONLY** on technical content, code, tutorials, actionable insights

**MANDATORY ARTICLE STRUCTURE** (exact Medium/LinkedIn format):
- Write in **first-person professional tone**
- Use **bold subheadings** and **numbered lists**
- Include **code snippets** for technical videos
- Make **Actionable Steps** copy-paste ready
- End with a **short summary of the article**

Transcript:
{transcript}
"""

summarizer_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(ARTICLE_SYSTEM),
    HumanMessagePromptTemplate.from_template(ARTICLE_HUMAN),
])


# ─────────────────────────────────────────────
# 3.  Transcript helpers
# ─────────────────────────────────────────────
def extract_transcript(link: str) -> str:
    """Download and return the raw transcript text from a YouTube URL."""
    loader = YoutubeLoader.from_youtube_url(link)
    docs = loader.load()
    return docs[0].page_content


def get_text_chunks(text: str, chunk_size: int = 5000, chunk_overlap: int = 200) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_text(text)


# ─────────────────────────────────────────────
# 4.  Recursive summariser (long transcripts)
# ─────────────────────────────────────────────
def recursive_summarize(text: str, llm) -> str:
    """Chunk a long transcript and build a rolling article summary."""
    chunks = get_text_chunks(text)
    running_summary = ""

    for i, chunk in enumerate(chunks):
        prompt = f"""
You are summarising technical content into a professional article.

Current summary so far:
{running_summary}

New content (chunk {i + 1}/{len(chunks)}):
{chunk}

**CRITICAL INSTRUCTIONS**:
- IGNORE intro notes, channel names, subscribe/like/comment prompts, marketing phrases
- FOCUS ONLY on technical content, code, tutorials, actionable insights

**MANDATORY ARTICLE STRUCTURE**:
- First-person professional tone
- Bold subheadings and numbered lists
- Code snippets where relevant
- Actionable, copy-paste ready steps
- Short summary at the end
"""
        response = llm.invoke(prompt)
        running_summary = response.content

    return running_summary


# ─────────────────────────────────────────────
# 5.  Webpage-generator prompt
# ─────────────────────────────────────────────
WEB_SYSTEM = """You are a Senior Frontend Web Developer with 10+ years of experience in HTML5, CSS3, and modern JavaScript (ES6+).

Your task: Generate COMPLETE, PRODUCTION-READY frontend code.

**MANDATORY OUTPUT FORMAT** (use these exact delimiters):
--html--
[html code here]
--html--

--css--
[css code here]
--css--

--js--
[javascript code here]
--js--
"""

WEB_HUMAN = """
Create a **production-ready article webpage** styled like **Medium, Dev.to, Hashnode, and Substack**.

**MANDATORY REQUIREMENTS**:
- Mobile-first responsive design (perfect on all devices)
- Clean, modern typography (readable system fonts)
- Medium-like article layout with card-based design
- Dark/light theme toggle
- Smooth animations and scroll effects
- SEO-optimised with proper meta tags
- Accessibility compliant (ARIA labels, keyboard navigation)

**CONTENT TO USE**:
{article_content}
"""

web_dev_template = ChatPromptTemplate.from_messages([
    ("system", WEB_SYSTEM),
    ("human", WEB_HUMAN),
])


# ─────────────────────────────────────────────
# 6.  Smart pipeline
# ─────────────────────────────────────────────
def build_pipeline(llm):
    def estimate_long(link: str) -> bool:
        transcript = extract_transcript(link)
        return len(transcript) >= 1000

    base_summarizer = (
        RunnablePassthrough()
        | RunnableLambda(extract_transcript)
        | summarizer_prompt
        | llm
        | StrOutputParser()
    )

    long_summarizer = (
        RunnablePassthrough()
        | RunnableLambda(extract_transcript)
        | RunnableLambda(lambda text: recursive_summarize(text, llm))
    )

    smart_summarizer = RunnableBranch(
        (RunnableLambda(estimate_long), long_summarizer),
        base_summarizer,
    )

    full_pipeline = smart_summarizer | web_dev_template | llm | StrOutputParser()
    return full_pipeline


# ─────────────────────────────────────────────
# 7.  File extraction helpers
# ─────────────────────────────────────────────
def extract_section(raw: str, tag: str) -> str:
    """Extract content between --tag-- delimiters."""
    parts = raw.split(f"--{tag}--")
    return parts[1].strip() if len(parts) >= 3 else ""


def build_zip(html: str, css: str, js: str) -> bytes:
    import io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("index.html", html)
        zf.writestr("style.css", css)
        zf.writestr("script.js", js)
    return buf.getvalue()


# ─────────────────────────────────────────────
# 8.  Streamlit UI
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube → Article Generator",
    page_icon="🎬",
    layout="wide",
)

# ── Custom CSS ──────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    padding: 3rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    color: white;
}
.hero h1 { font-size: 2.8rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
.hero p  { font-size: 1.1rem; opacity: 0.8; margin-top: 0.5rem; }

.step-badge {
    display: inline-block;
    background: #e8f4fd;
    color: #0f3460;
    border-radius: 20px;
    padding: 0.25rem 0.8rem;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 0.4rem;
}

.result-card {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 12px;
    padding: 1.5rem;
    margin-top: 1rem;
}

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: opacity 0.2s;
}
div[data-testid="stButton"] > button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)

# ── Hero banner ─────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🎬 YouTube → Article</h1>
  <p>Paste a YouTube link and get a polished, publish-ready article webpage — powered by Gemini.</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar – API key ────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Get your free key at https://aistudio.google.com/app/apikey",
    )
    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown("""
1. 🔑 Enter your Gemini API key  
2. 🔗 Paste a YouTube URL  
3. ✨ Click Generate  
4. 📥 Download the webpage files  
    """)
    st.markdown("---")
    st.caption("Built with LangChain + Gemini 2.0 Flash")

# ── Main content ─────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<span class="step-badge">STEP 1</span>', unsafe_allow_html=True)
    youtube_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed",
    )

with col2:
    st.markdown('<span class="step-badge">STEP 2</span>', unsafe_allow_html=True)
    generate_btn = st.button("✨ Generate Article", use_container_width=True)

# ── Generation logic ─────────────────────────
if generate_btn:
    if not api_key:
        st.error("🔑 Please enter your Google Gemini API key in the sidebar.")
    elif not youtube_url or "youtube.com" not in youtube_url and "youtu.be" not in youtube_url:
        st.warning("⚠️ Please enter a valid YouTube URL.")
    else:
        with st.spinner("🎬 Fetching transcript…"):
            try:
                transcript_preview = extract_transcript(youtube_url)
                word_count = len(transcript_preview.split())
                st.success(f"✅ Transcript loaded — **{word_count:,} words**")
            except Exception as e:
                st.error(f"❌ Could not fetch transcript: {e}")
                st.stop()

        with st.spinner("✍️ Writing article with Gemini…"):
            try:
                llm = get_llm(api_key)
                pipeline = build_pipeline(llm)
                raw_output = pipeline.invoke(youtube_url)
            except Exception as e:
                st.error(f"❌ Generation failed: {e}")
                st.stop()

        # ── Parse output sections ────────────────
        html_code = extract_section(raw_output, "html")
        css_code  = extract_section(raw_output, "css")
        js_code   = extract_section(raw_output, "js")

        if not html_code:
            st.warning("⚠️ Could not parse HTML from model output. Showing raw output below.")
            st.code(raw_output, language="markdown")
        else:
            st.success("🎉 Article webpage generated successfully!")

            # ── Tabs: Preview + Code ─────────────
            tab1, tab2, tab3, tab4 = st.tabs(["🌐 Preview", "📄 HTML", "🎨 CSS", "⚡ JavaScript"])

            with tab1:
                full_page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Generated Article</title>
<style>{css_code}</style>
</head>
<body>
{html_code}
<script>{js_code}</script>
</body>
</html>"""
                st.components.v1.html(full_page, height=700, scrolling=True)

            with tab2:
                st.code(html_code, language="html")

            with tab3:
                st.code(css_code, language="css")

            with tab4:
                st.code(js_code, language="javascript")

            # ── Downloads ────────────────────────
            st.markdown("---")
            st.subheader("📥 Download Files")

            dl1, dl2, dl3, dl4 = st.columns(4)

            with dl1:
                st.download_button(
                    "⬇️ index.html",
                    data=html_code,
                    file_name="index.html",
                    mime="text/html",
                    use_container_width=True,
                )
            with dl2:
                st.download_button(
                    "⬇️ style.css",
                    data=css_code,
                    file_name="style.css",
                    mime="text/css",
                    use_container_width=True,
                )
            with dl3:
                st.download_button(
                    "⬇️ script.js",
                    data=js_code,
                    file_name="script.js",
                    mime="text/javascript",
                    use_container_width=True,
                )
            with dl4:
                zip_bytes = build_zip(html_code, css_code, js_code)
                st.download_button(
                    "📦 website.zip",
                    data=zip_bytes,
                    file_name="website.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
