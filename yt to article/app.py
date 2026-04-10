import os
import io
import zipfile
import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF
import re

from langchain_community.document_loaders import YoutubeLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# -----------------------------
# Load ENV
# -----------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in .env file")
    st.stop()

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="YT → Article & PDF",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 YouTube → Article & PDF Generator")
st.caption("Paste a YouTube URL → Get article + PDF + webpage")

# -----------------------------
# Input
# -----------------------------
youtube_url = st.text_input("🔗 YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
run_btn = st.button("⚡ Generate Article", use_container_width=True)

# -----------------------------
# Chains
# -----------------------------
def get_chains():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.7,
    )

    article_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a Professional Article Writer."
        ),
        HumanMessagePromptTemplate.from_template("""
Convert this YouTube transcript into a professional blog article.

IGNORE:
- welcome messages
- subscribe lines
- promotions

Focus only on useful content.

Structure:
- Headings
- Bullet points
- Clean explanation

TRANSCRIPT:
{transcript}
""")
    ])

    webpage_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("""
Generate COMPLETE frontend code.

FORMAT:
--html--
...
--html--

--css--
...
--css--

--js--
...
--js--
"""),
        HumanMessagePromptTemplate.from_template("""
Create a clean modern article webpage.

ARTICLE:
{article_content}
""")
    ])

    summarizer = (
        RunnableLambda(lambda url: YoutubeLoader.from_youtube_url(url).load()[0].page_content)
        | RunnableLambda(lambda t: {"transcript": t})
        | article_prompt
        | llm
        | StrOutputParser()
    )

    webpage = (
        RunnableLambda(lambda a: {"article_content": a})
        | webpage_prompt
        | llm
        | StrOutputParser()
    )

    return summarizer, webpage

# -----------------------------
# Parse Web Output
# -----------------------------
def parse_output(raw):
    def extract(tag):
        try:
            return raw.split(f"--{tag}--")[1].strip()
        except:
            return ""
    return extract("html"), extract("css"), extract("js")

# -----------------------------
# PDF Generator
# -----------------------------
def generate_pdf(article_text):
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "YouTube Article Summary", align="C")
    pdf.ln(10)

    # Content
    pdf.set_font("Arial", size=10)

    clean = re.sub(r"```.*?```", "", article_text, flags=re.DOTALL)
    clean = clean.replace("**", "").replace("#", "").replace("`", "")

    for line in clean.split("\n"):
        line = line.strip()
        if not line:
            pdf.ln(3)
            continue

        try:
            pdf.multi_cell(0, 6, line)
        except:
            continue

    return pdf.output(dest='S').encode('latin-1')

# -----------------------------
# Run App
# -----------------------------
if run_btn:
    if not youtube_url:
        st.error("❌ Please enter YouTube URL")
    else:
        try:
            summarizer, webpage_chain = get_chains()

            with st.spinner("🧠 Generating article..."):
                article = summarizer.invoke(youtube_url)

            st.success("✅ Article generated!")

            # Article Display
            st.subheader("📄 Article")
            st.markdown(article)

            # PDF Download
            pdf_bytes = generate_pdf(article)
            st.download_button(
                "📥 Download PDF",
                pdf_bytes,
                "article.pdf",
                mime="application/pdf"
            )

            # Webpage Generation
            with st.spinner("🌐 Generating webpage..."):
                raw = webpage_chain.invoke(article)

            html, css, js = parse_output(raw)

            full_html = f"""
            <html>
            <head><style>{css}</style></head>
            <body>{html}<script>{js}</script></body>
            </html>
            """

            st.components.v1.html(full_html, height=600)

            # ZIP Download
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as z:
                z.writestr("index.html", html)
                z.writestr("style.css", css)
                z.writestr("script.js", js)

            zip_buffer.seek(0)

            st.download_button(
                "⬇️ Download Website",
                zip_buffer,
                "website.zip",
                mime="application/zip"
            )

        except Exception as e:
            st.error(f"❌ Error: {e}")