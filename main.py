from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import boto3
import json
import pdfplumber
import tempfile
import os

app = FastAPI()

# AWS Bedrock Setup
bedrock = boto3.client("bedrock-runtime", region_name="ap-south-1")
inference_profile_arn = "arn:aws:bedrock:ap-south-1:069717477936:inference-profile/apac.amazon.nova-micro-v1:0"

# Extract text from PDF
def extract_pdf_text(file_path):
    with pdfplumber.open(file_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

# Clean input
def clean_text(text, max_len=4000):
    return text.strip().replace('\n', ' ')[:max_len]

# Build prompt with improved structure
def build_prompt(typed_text, extracted_text):
    return f"""
You are tasked as a business analyst conducting deep research on innovative companies. Your job is to analyze product information and return comprehensive insights as JSON.

**CONTENT TO ANALYZE:**
Founder Notes: {typed_text}
Pitch Content: {clean_text(extracted_text)}

**REQUIRED JSON OUTPUT:**
{{
  "title": "Product name with key differentiator",
  "description": "Detailed 3-4 sentence explanation of what the product does, target market, and quantifiable impact",
  "audience": "Return a concise, comma-separated list of specific user types relevant to the product — such as roles, industries, or customer segments — based only on the input content. Do not generate paragraphs or explanations.",
  "problemStatements": [
    "Articulate a significant problem statement unique to the domain, considering nuances and impacts.",
    "Define another distinct problem that addresses a different dimension with real-world implications.", 
    "Develop a third problem statement focusing on another aspect, bringing forward contextual challenges."
  ],
  "tags": ["Identify technical and business keywords drawn from content specifics"],
  "followUpQuestions": [
    "Derive a question regarding an intriguing unique aspect of the company, emphasizing why it matters.",
    "Formulate a differentiated inquiry about another distinctive characteristic, showcasing deep understanding.",
    "Propose a distinct question exploring an aspect that only this company could authentically answer."
  ],
  "burningProblems": [
    "Acknowledgment of a current business challenge grounded in their market position or business stage.",
    "Assessment of a pressing issue pertinent to their operational, growth, and strategic landscapes.", 
    "Declaration of a realistic business challenge with immediate practical implications."
  ]
}}

**DETAILED ANALYSIS REQUIREMENTS:**

**PROBLEM STATEMENTS (2-3 sentences each):**
Ensure problem statements are derived from actual content, reflecting specific sector challenges and impacts. Articulate who the problems affect, why they matter, supported by quantifiable data where possible.

**FOLLOW-UP QUESTIONS:**
Identify 3 genuinely unique aspects of the content that warrant exploration. Craft questions that reveal insightful understanding of those specific facets.

QUESTION CREATION RULES:
- Focus on intriguing, detailed company-specific information.
- Explore implementation, rationale, or impacts of these aspects.
- Formulate structurally distinct questions relevant only to this company.
- Keep curiosity grounded in demonstrated facts.

**BURNING PROBLEMS (business challenge statements):**
Assess the business landscape and extract 3 genuine challenges being faced. Address immediate, identifiable obstacles with clarity, focusing on their business dimensions.

CHALLENGE IDENTIFICATION:
- Consider their market position, business stage, and sector context.
- Highlight current operational, growth, and strategic challenges.
- Make challenges specific and relevant without relying on general templates.

**QUALITY STANDARDS:**
- Output must reflect detailed understanding based on specific input content.
- Avoid generic language; adapt to content-specific terminology and context.
- Content should be substantial and reflective of the unique business situation.

**CONTENT EXTRACTION RULES:**
- Ensure extraction is true to the input specifics, including technologies, customer segments, and terminology.
- Avoid adding information not present in the input.

**CRITICAL FORMATTING:**
- Arrays should contain exactly 3 string items per requirement.
- Include fully developed content for each item with no single-sentence responses.
- Ensure outputs are comprehensive, specific, and tailored to the company.

**FINAL CHECK:**
Verify outputs reflect an understanding of a unique, specific business with distinct challenges. Generic outputs applicable to multiple companies are unacceptable.

Respond with valid JSON only.
"""

# Call Nova Micro via Bedrock
def query_nova_micro(prompt_text):
    body = {
        "inferenceConfig": {
            "max_new_tokens": 1200,
            "temperature": 0.7
        },
        "messages": [
            {
                "role": "user",
                "content": [{"text": prompt_text}]
            }
        ]
    }

    response = bedrock.invoke_model_with_response_stream(
        modelId=inference_profile_arn,
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

# Main API Route
@app.post("/idea-capture")
async def capture_idea(
    typed_input: str = Form(...),
    file: UploadFile = File(...)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        file_path = tmp.name
        tmp.write(await file.read())

    try:
        extracted_text = extract_pdf_text(file_path)
        prompt = build_prompt(typed_input, extracted_text)
        raw_output = query_nova_micro(prompt)

        try:
            parsed = json.loads(raw_output)
            return JSONResponse(content=parsed)
        except json.JSONDecodeError:
            return JSONResponse(
                content={"error": "LLM returned unparseable output", "raw": raw_output},
                status_code=200
            )

    finally:
        os.remove(file_path)