import streamlit as st
import boto3
import json
import pdfplumber
import tempfile
import os
import re
import typing as t
import anthropic
from openai import OpenAI


# ---------- AWS Bedrock Setup ----------
bedrock = boto3.client("bedrock-runtime", region_name="ap-south-1")
nova_inference_arn = (
    "arn:aws:bedrock:ap-south-1:069717477936:inference-profile/apac.amazon.nova-pro-v1:0"
)

# ---------- Helper: Extract PDF Text & Highlights ----------
def extract_pdf_text(file_path: str) -> tuple[str, str]:
    full_text, highlights = [], []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
            text = page.extract_text() or ""
            full_text.append(text)

            for word in words:
                if float(word.get("size", 0)) >= 16 or "bold" in word.get("fontname", "").lower():
                    highlights.append(word["text"])

    return "\n".join(full_text), "\n".join(set(highlights))


def clean_text(text: str, max_len: int = 4000) -> str:
    return text.strip().replace("\n", " ")[:max_len]


# ---------- Modified Prompt Builder ----------
def build_prompt(typed_text: str, extracted_text: str, highlighted_text: str, founder_goal: str) -> str:
    return f"""
You are an expert startup analyst. Analyze this startup comprehensively using the 10 Guard Rails framework, then generate targeted insights ONLY for genuine gaps.

--- MATERIALS ---
Pitch Deck: {clean_text(extracted_text)}
Highlights: {highlighted_text}
Founder Input: {typed_text}
Assistance Goal: {founder_goal}

--- ANALYSIS FRAMEWORK: 10 GUARD RAILS ---
1. VISION: Societal impact and long-term purpose
2. TARGET CUSTOMER: Buyer personas and user segments  
3. VALUE PROPOSITION: Quantified benefits and outcomes
4. TECHNICAL: Core technology and architecture
5. UNIQUE VALUE: Defensible competitive moats
6. MARKETING: Customer acquisition strategy
7. REVENUE MODEL: Monetization and unit economics
8. COMPETITIVE LANDSCAPE: Market alternatives and positioning
9. OPERATIONS & SCALABILITY: Implementation and delivery process
10. LEGAL & REGULATORY: Compliance framework and IP protection

--- MANDATORY ASSESSMENT PROCESS ---
Before generating output, internally evaluate each guard rail:

✅ WELL-COVERED = Specific details, metrics, clear strategy provided
❌ WEAK/MISSING = Vague mentions, no specifics, or completely absent

**RULE: Generate follow-ups and burning problems ONLY for ❌ WEAK/MISSING areas**

Examples of what NOT to ask if well-covered:
- Don't ask "What's your revenue model?" if SaaS subscription is clearly stated
- Don't ask "Who's your target market?" if specific industries/roles are mentioned
- Don't ask "What's your tech stack?" if architecture is detailed

--- OUTPUT FORMAT (JSON) ---
{{
  "title": "Product name with unique positioning",
  "description": "3-4 sentences using actual metrics and specifics from materials",
  "audience": "Precise roles/segments mentioned in content",
  "problemStatements": ["3 specific pain points with business impact"],
  "tags": ["Industry/domain themes from content"],
  "followUpQuestions": ["Exactly 3 analytical, context-rich questions (2-3 sentences each)"],
  "burningProblems": ["Exactly 3 business risks from actual gaps identified"]
}}

--- FOLLOW-UP QUESTION REQUIREMENTS ---
Each question must be:
- **2-3 sentences long** with analytical depth
- **Context-aware**: Reference specific details from their materials
- **Strategic**: Address complex business challenges, not basic facts
- **Sophisticated**: Show deep understanding of their industry/situation

AVOID: Simple, direct questions like "What is your target market?"
PREFER: Complex, analytical questions like "Given your focus on X industry and the challenges of Y regulatory environment, how are you planning to navigate the inherent tension between rapid market entry and the extensive validation requirements that institutional buyers typically demand?"
"""


# ---------- Claude (Anthropic) ----------
def query_claude(prompt: str) -> str:
    """
    Calls Claude 3.5 Haiku. Tries strict JSON mode; if the
    installed SDK is too old for `format="json"`, it retries
    without it and relies on the extractor.
    """
    client = anthropic.Anthropic(api_key=st.secrets["anthropic"]["api_key"])
    model_id = "claude-3-5-haiku-20241022"          # <-- correct ID

    try:
        # Preferred path: SDK >= 0.21.0 supports `format="json"`
        response = client.messages.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
            format="json",                          # <-- strict mode
        )
    except TypeError:
        # Older SDK fallback (no `format` kwarg)
        response = client.messages.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
        )

    # Anthropic returns a list of content blocks
    return response.content[0].text if response.content else ""



def query_nova_pro(prompt: str) -> str:
    """
    Nova Pro via AWS Bedrock – unchanged (already streams plain text).
    """
    body = {
        "inferenceConfig": {"max_new_tokens": 1500, "temperature": 0.3},
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
    }

    response = bedrock.invoke_model_with_response_stream(
        modelId=nova_inference_arn,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
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


client = OpenAI(api_key=st.secrets["openai"]["api_key"])

def query_gpt(prompt: str) -> str:
    """
    GPT‑4 Turbo with explicit JSON response_format.
    """
    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1500,
        response_format={"type": "json_object"},  # <-- forces JSON
    )
    return response.choices[0].message.content


# ---------- Robust JSON Extractor ----------
def extract_json(raw: str) -> t.Optional[dict]:
    """
    Best‑effort extraction when a model still slips extra prose or markdown fences.
    """
    cleaned = raw.strip()

    # Remove ```json … ``` or ``` … ``` fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9]*\s*", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned).strip()

    # First attempt direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: grab first {...} block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Outlaw Idea Capture", layout="wide")
st.title("📊 Outlaw Idea Capture AI")
st.markdown("Upload your pitch deck and brief notes to extract structured business insights.")

typed_input = st.text_area("📝 Enter Founder Notes / Product Description", height=200)
founder_goal = st.text_area("🎯 What help do you want from Outlaw?", height=100)
uploaded_file = st.file_uploader("📎 Upload Pitch Deck (PDF)", type=["pdf"])
model_choice = st.selectbox(
    "🤖 Choose Model",
    ["Nova Pro (AWS)", "Claude 3.5 Haiku (Anthropic)", "GPT-4 Turbo (OpenAI)"],
    index=0,
)

if st.button("🔍 Analyze"):
    if not uploaded_file or not typed_input or not founder_goal:
        st.error("Please provide founder notes, a pitch deck, and a goal.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name

        with st.spinner("Extracting insights..."):
            extracted_text, highlighted = extract_pdf_text(file_path)
            prompt = build_prompt(typed_input, extracted_text, highlighted, founder_goal)

            if model_choice == "Nova Pro (AWS)":
                raw_output = query_nova_pro(prompt)
            elif model_choice == "Claude 3.5 Haiku (Anthropic)":
                raw_output = query_claude(prompt)
            else:
                raw_output = query_gpt(prompt)

            parsed = extract_json(raw_output)
            if parsed is not None:
                st.success("✅ Insights generated successfully.")
                st.json(parsed)
            else:
                st.warning("⚠️ Output could not be parsed as JSON. Showing raw output below.")
                st.code(raw_output)

        os.remove(file_path)