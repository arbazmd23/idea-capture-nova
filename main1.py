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

# Build prompt with domain-agnostic approach
def build_prompt(typed_text, extracted_text):
    return f"""
Analyze this startup as if you're meeting the founder for coffee and genuinely curious about what they're building.

**FOUNDER NOTES:**
{typed_text}

**PITCH DECK CONTENT:**
{clean_text(extracted_text)}

Your goal: Understand their world deeply enough to ask questions that make them think "Wow, this person really gets what I'm trying to do."

Think about:
- What's truly unique about their approach?
- What assumptions are they making that might be wrong?
- What details did they skip over that seem important?
- What would worry you if you were in their shoes?
- What would you be curious about if this was your friend's startup?

Generate follow-up questions that show you understand their specific context. Each question should make the founder pause and think - not give rehearsed answers they've given a hundred times before.

For burning problems, think about what specifically stresses THIS founder out. Not what stresses all founders, but what keeps THIS person awake based on what they're actually building and the world they're operating in.

Be intellectually curious. Use your understanding of their domain, technology, market, and situation to generate insights that feel personal and relevant to their journey.

Output ONLY valid JSON:

{{
  "title": "Product name and positioning",
  "description": "Clear explanation showing you understand what they're building and why it matters",
  "audience": "Who will actually pay for this product",
  "problemStatements": [
    "Three specific problems this product addresses"
  ],
  "tags": [
    "5-8 relevant tags about the product/technology/domain"
  ],
  "followUpQuestions": [
    "Three questions that demonstrate deep understanding of their specific situation"
  ],
  "burningProblems": [
    "Three specific challenges THIS founder likely faces based on their context"
  ]
}}

Trust your intelligence. Be genuinely curious about their specific situation."""

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