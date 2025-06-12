# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.responses import JSONResponse
# import anthropic
# import pdfplumber
# import tempfile
# import os
# import json

# app = FastAPI()

# claude_client = anthropic.Anthropic(api_key=st.secrets["anthropic"]["api_key"])


# # Extract text + high-importance lines from PDF
# def extract_pdf_text(file_path):
#     full_text = []
#     highlights = []

#     with pdfplumber.open(file_path) as pdf:
#         for page in pdf.pages:
#             words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
#             page_text = page.extract_text() or ""
#             full_text.append(page_text)

#             # Capture headlines / bold text / larger fonts
#             for word in words:
#                 if float(word.get("size", 0)) >= 16 or word.get("fontname", "").lower().startswith("bold"):
#                     highlights.append(word["text"])

#     return "\n".join(full_text), "\n".join(set(highlights))

# # Clean input
# def clean_text(text, max_len=4000):
#     return text.strip().replace('\n', ' ')[:max_len]

# def build_prompt(typed_text, extracted_text, highlighted_text):
#     return f"""
# You are an expert business analyst helping validate early-stage startup ideas.

# CONTENT PROVIDED:
# - Founder Notes: {typed_text}
# - Full Pitch Text: {clean_text(extracted_text)}
# - Key Highlights / Headers from Pitch: {highlighted_text}

# TASK: Analyze the above and return structured business insights in JSON.

# RULES:
# - Extract and prioritize meaningful **headlines**, **key metrics**, and **bolded/highlighted content** from the pitch.
# - Pay attention to **finance, marketing, business model, legal/compliance, growth strategy, and operations** — not just product/tech.
# - Insights must reflect what’s actually in the content. Avoid generic assumptions.

# OUTPUT FORMAT (respond with valid JSON only):

# {{
#   "title": "Product name with key differentiator",
#   "description": "3–4 sentence explanation of what the product does, for whom, and why it matters — using real metrics or content if available",
#   "audience": "Comma-separated list of specific roles, users, or customer types derived from content — no full sentences",
#   "problemStatements": [
#     "...",
#     "...",
#     "..."
#   ],
#   "tags": ["...", "...", "..."],
#   "followUpQuestions": [
#     "...",
#     "...",
#     "..."
#   ],
#   "burningProblems": [
#     "...",
#     "...",
#     "..."
#   ]
# }}

# REQUIREMENTS FOR EACH SECTION:

# **PROBLEM STATEMENTS (2–3 sentences each):**
# - Must reflect nuanced, domain-specific pain points described or implied in the content.
# - Each should highlight who is affected and what consequences arise — with real-world or measurable impact if possible.
# - Avoid broad or universal startup issues unless they’re clearly described.

# **FOLLOW-UP QUESTIONS:**
# - Should reflect genuinely curious, contextual questions derived from the startup’s unique details.
# - Focus on specifics (metrics, claims, partnerships, go-to-market, differentiation).
# - Avoid generic or early-stage questions if not implied.

# **BURNING PROBLEMS:**
# - These are actual current business challenges inferred from the startup’s position, stage, or model.
# - Can relate to business growth, scale, finance, hiring, compliance, or operational difficulties.
# - Each one should be distinct and clearly justified by the content.

# **QUALITY STANDARDS:**
# - Use only what’s found in the pitch or typed input.
# - Outputs must be grounded, original, and contextual.
# - Avoid generic templates, AI clichés, or surface-level speculation.

# FORMAT:
# - JSON only
# - Each array must have exactly 3 fully developed entries.
# - All entries must be tailored to the input, with no placeholders.

# Before generating, ensure the output reflects an understanding of a **real, specific company** with **real challenges and content** — not a hypothetical startup.
# """


# # Query Claude
# def query_claude(prompt_text):
#     response = client.messages.create(
#         model="claude-3-5-haiku-20241022",
#         max_tokens=1500,
#         messages=[{"role": "user", "content": prompt_text}]
#     )
#     return response.content[0].text if response.content else ""

# # API Endpoint
# @app.post("/idea-capture")
# async def capture_idea(
#     typed_input: str = Form(...),
#     file: UploadFile = File(...)
# ):
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#         file_path = tmp.name
#         tmp.write(await file.read())

#     try:
#         full_text, highlighted = extract_pdf_text(file_path)
#         prompt = build_prompt(typed_input, full_text, highlighted)
#         raw_output = query_claude(prompt)

#         try:
#             parsed = json.loads(raw_output)
#             return JSONResponse(content=parsed)
#         except json.JSONDecodeError:
#             return JSONResponse(
#                 content={"error": "LLM returned unparseable output", "raw": raw_output},
#                 status_code=200
#             )

#     finally:
#         os.remove(file_path)
