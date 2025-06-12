import streamlit as st
import boto3
import json
import pdfplumber
import tempfile
import os
import anthropic

# === AWS Bedrock Setup ===
bedrock = boto3.client("bedrock-runtime", region_name="ap-south-1")
nova_inference_arn = "arn:aws:bedrock:ap-south-1:069717477936:inference-profile/apac.amazon.nova-pro-v1:0"

# === PDF Extraction ===
def extract_pdf_text(file_path):
    full_text = []
    highlights = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
            text = page.extract_text() or ""
            full_text.append(text)

            for word in words:
                if float(word.get("size", 0)) >= 16 or "bold" in word.get("fontname", "").lower():
                    highlights.append(word["text"])

    return "\n".join(full_text), "\n".join(set(highlights))

def clean_text(text, max_len=4000):
    return text.strip().replace('\n', ' ')[:max_len]

def build_prompt(typed_text, extracted_text, highlighted_text):
    return f"""
You are an expert business analyst helping validate early-stage startup ideas.

CONTENT PROVIDED:
- Founder Notes: {typed_text}
- Full Pitch Text: {clean_text(extracted_text)}
- Key Highlights / Headers from Pitch: {highlighted_text}

TASK: Analyze the above and return structured business insights in JSON.

RULES:
- Give high weight to bolded or large-font text (headlines), especially if they include metrics, claims, or positioning statements.
- Always extract and elevate meaningful quantitative or strategic information from headers and key sentences.
- Pay attention to **finance, marketing, business model, legal/compliance, growth strategy, operations, and product** — not just technical aspects.
- Avoid assumptions; stick to the content.

OUTPUT FORMAT (respond with valid JSON only):

{{
  "title": "Product name with key differentiator",
  "description": "3–4 sentence explanation of what the product does, for whom, and why it matters — using real metrics or content if available",
  "audience": "Comma-separated list of specific roles, users, or customer types derived from content — no full sentences",
  "problemStatements": [
    "...",
    "...",
    "..."
  ],
  "tags": ["...", "...", "..."],
  "followUpQuestions": [
    "...",
    "...",
    "..."
  ],
  "burningProblems": [
    "...",
    "...",
    "..."
  ]
}}

REQUIREMENTS FOR EACH SECTION:

**PROBLEM STATEMENTS (2–3 sentences each):**
- Must reflect nuanced, domain-specific pain points described or implied in the content.
- Each should highlight who is affected and what consequences arise — with real-world or measurable impact if possible.

**FOLLOW-UP QUESTIONS:**
- Derive 3 original questions that reflect curiosity about this specific startup.
- At least one must be business-related (finance, GTM, ops, legal, compliance, or marketing strategy).
- Avoid early-stage clichés or generic curiosity.
- Do not include multiple tech-only questions.

**BURNING PROBLEMS:**
- Identify 3 urgent business challenges the company is likely facing based on the content.
- These can involve funding, team building, compliance, market entry, scalability, partnerships, or model limitations.
- Use realistic, content-grounded framing — avoid hypotheticals.

**QUALITY STANDARDS:**
- Use only what’s found in the pitch or typed input.
- Outputs must be grounded, original, and contextual.
- Avoid generic templates, AI clichés, or surface-level speculation.

FORMAT:
- JSON only
- Each array must have exactly 3 fully developed entries.
- All entries must be tailored to the input, with no placeholders.

Before generating, ensure the output reflects an understanding of a **real, specific company** with **real challenges and content** — not a hypothetical startup.
"""


# === Claude Haiku via Anthropic ===
def query_claude(prompt):
    client = anthropic.Anthropic(api_key=st.secrets["anthropic"]["api_key"])
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text if response.content else ""

# === Nova Pro via AWS Bedrock ===
def query_nova_pro(prompt):
    body = {
        "inferenceConfig": {
            "max_new_tokens": 1500,
            "temperature": 0.3
        },
        "messages": [
            {"role": "user", "content": [{"text": prompt}]}
        ]
    }

    response = bedrock.invoke_model_with_response_stream(
        modelId=nova_inference_arn,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )

    output_string = ""
    for event in response["body"]:
        if "chunk" in event:
            chunk = event["chunk"]["bytes"]
            if chunk:
                try:
                    payload = json.loads(chunk.decode("utf-8"))
                    if "contentBlockDelta" in payload:
                        output_string += payload["contentBlockDelta"]["delta"].get("text", "")
                except Exception:
                    continue
    return output_string

# === Streamlit App ===
st.set_page_config(page_title="Outlaw Idea Capture", layout="wide")
st.title("📊 Outlaw Idea Capture AI")
st.markdown("Upload your pitch deck (PDF) and brief notes to extract structured insights.")

typed_input = st.text_area("📝 Enter Founder Notes / Product Description", height=200)
uploaded_file = st.file_uploader("📎 Upload Pitch Deck (PDF)", type=["pdf"])

model_choice = st.selectbox("🤖 Choose Model", ["Nova Pro (AWS)", "Claude 3.5 Haiku (Anthropic)"], index=0)

if st.button("🔍 Analyze"):
    if not uploaded_file or not typed_input:
        st.error("Please provide both founder notes and a pitch deck.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name

        with st.spinner("Extracting insights..."):
            extracted_text, highlighted = extract_pdf_text(file_path)
            prompt = build_prompt(typed_input, extracted_text, highlighted)

            if model_choice == "Nova Pro (AWS)":
                raw_output = query_nova_pro(prompt)
            else:
                raw_output = query_claude(prompt)

            try:
                parsed = json.loads(raw_output)
                st.success("✅ Insights generated successfully.")
                st.json(parsed)
            except json.JSONDecodeError:
                st.warning("⚠️ Output could not be parsed as JSON. Showing raw output below.")
                st.code(raw_output)

        os.remove(file_path)
