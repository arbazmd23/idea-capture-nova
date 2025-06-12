from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import boto3
import json
import pdfplumber
import tempfile
import os
import re

app = FastAPI()

# AWS Bedrock Setup
bedrock = boto3.client("bedrock-runtime", region_name="ap-south-1")
inference_profile_arn = "arn:aws:bedrock:ap-south-1:069717477936:inference-profile/apac.amazon.nova-micro-v1:0"

# === Utility Functions ===
def extract_pdf_text(file_path):
    with pdfplumber.open(file_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def clean_text(text, max_len=4000):
    return text.strip().replace('\n', ' ')[:max_len]
# === V8 DYNAMIC CONTENT-DRIVEN ANALYSIS PROMPT ===
def build_analysis_prompt(typed_text: str, extracted_text: str) -> str:
    return f"""
Analyze this product information and return insights as JSON.

**INPUT:**
Founder Notes: {typed_text}
Pitch Content: {clean_text(extracted_text)}

**OUTPUT (JSON only):**
{{
  "title": "Product name with key differentiator that captures the core innovation or unique positioning",
  "description": "A compelling 2-3 sentence explanation that clearly articulates what the product does, the specific target market it serves, and the quantifiable impact or transformation it delivers. Highlight the core technology and unique value proposition that sets it apart from existing solutions.",
  "audience": "The primary user/buyer segment with specific details about their roles, industry context, current pain points, and what they specifically value in a solution. Include their decision-making criteria and typical operational challenges.",
  "problemStatements": [
    "Extract and articulate the first major pain point or inefficiency that this specific product directly addresses. Base this entirely on what's mentioned or implied in the input content. Include quantifiable impacts, specific industry context, or measurable consequences where available.",
    "Extract and articulate the second distinct challenge that this product solves. Focus on different aspects of the problem space mentioned in the content. Include specific details about how this manifests for their target users.",
    "Extract and articulate the third problem area this product tackles. Ensure this addresses a different dimension of their solution space. Include context about why this problem matters specifically to their audience."
  ],
  "tags": ["Domain-specific keywords derived directly from the content, including core technologies, industry verticals, business model type, competitive advantages, and unique technical approaches mentioned."],
  "followUpQuestions": [
    "First unique question based on specific content details",
    "Second unique question based on different specific content details", 
    "Third unique question based on different specific content details"
  ],
  "burningProblems": [
    "First realistic business challenge statement",
    "Second realistic business challenge statement",
    "Third realistic business challenge statement"
  ]
}}

**FOLLOW-UP QUESTIONS APPROACH:**

Read the input content thoroughly and identify the most specific, unique, and interesting details mentioned. For each detail, create a question that explores that specific aspect in depth. The questions should sound natural and conversational, as if asked by someone genuinely curious about the unique aspects of their business.

**QUESTION CREATION PROCESS:**
1. Scan the content for specific technologies, methodologies, numbers, processes, customer segments, or unique approaches
2. Pick the 3 most specific and unique details that would be interesting to explore further
3. For each detail, create a question that naturally explores that specific aspect
4. Each question should be completely different in structure and focus
5. Questions should feel like they come from someone who carefully read and understood the content
6. Avoid any repetitive patterns or similar question structures

**QUESTION QUALITY CHECKS:**
- Does this question reference something specific and unique from their content?
- Would this exact question be impossible to ask about a different company?
- Does it explore a genuinely interesting aspect of their approach?
- Is it different in structure and focus from the other questions?

**BURNING PROBLEMS APPROACH:**

Analyze the business context, stage, market, and operational details mentioned in the content. Based on this analysis, identify three realistic business challenges they are likely managing right now. Write these as clear, factual statements about what they are working on or dealing with.

**PROBLEM IDENTIFICATION PROCESS:**
1. Consider their current business stage and market position
2. Think about typical challenges for their industry and business model
3. Factor in their specific customer base and operational approach
4. Identify immediate, practical challenges they would be managing
5. Write each as a straightforward statement about their current situation

**PROBLEM STATEMENT GUIDELINES:**
- Write as factual statements about what they are likely managing or working on
- Focus on immediate, practical business challenges
- Avoid overly technical assumptions
- Keep statements realistic and grounded in common business challenges
- Each problem should address a different aspect of their business (operations, growth, market, etc.)
- Make problems specific to their situation but not overly complex

**CONTENT ANALYSIS RULES:**

Extract information directly from the provided content. Do not add information not present in the input. Focus on what makes this specific business unique based on their content.

For each section:
- **problemStatements**: Extract actual problems mentioned or clearly implied in the content
- **followUpQuestions**: Create questions about the most specific and unique aspects mentioned
- **burningProblems**: Infer realistic challenges based on their described business context

**CRITICAL REQUIREMENTS:**
- All questions must be completely different in structure and content
- All problems must be written as statements, not questions
- Focus on what makes this business unique based on their specific content
- Avoid generic business terminology unless specific to their content
- Ensure each element would be completely different for different businesses
- **CRITICAL**: Ensure 'problemStatements', 'followUpQuestions', and 'burningProblems' are always lists of exactly three separate string items

**UNIQUENESS TEST:**
Before finalizing the output, verify that if you were given completely different input content from a different domain, your questions and problems would be entirely different. If they wouldn't be, revise to make them more specific to this particular business.
"""

# === Bedrock Query Function ===
def query_nova_micro(prompt_text: str) -> str:
    body = {
        "inferenceConfig": {
            "max_new_tokens": 1000,
            "temperature": 0.4
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

def extract_json_from_response(response: str) -> dict:
    """Extract JSON from LLM response"""
    try:
        return json.loads(response.strip())
    except:
        # Find JSON in response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass
        
        # Return minimal structure if parsing fails
        return {
            "title": "Product Analysis",
            "description": "Analysis could not be completed",
            "audience": "",
            "problemStatements": [],
            "tags": [],
            "followUpQuestions": [],
            "burningProblems": []
        }

# === MAIN API ===
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
        prompt = build_analysis_prompt(typed_input, extracted_text)
        response = query_nova_micro(prompt)
        result = extract_json_from_response(response)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        return JSONResponse(
            content={"error": f"Analysis failed: {str(e)}"}, 
            status_code=500
        )
    
    finally:
        os.remove(file_path)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}