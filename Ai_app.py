import streamlit as st
import anthropic
import json
import re
import typing as t
from pydantic import BaseModel
import base64

# Set page config
st.set_page_config(
    page_title="Outlaw Idea Capture",
    page_icon="🚀",
    layout="wide"
)

# Response model (keeping the same structure)
class AnalysisResponse(BaseModel):
    title: str
    description: str
    audience: str
    problemStatements: list[str]
    tags: list[str]
    followUpQuestions: list[str]
    burningProblems: list[str]

@st.cache_resource
def get_anthropic_client():
    """Initialize Anthropic client using Streamlit secrets"""
    try:
        api_key = st.secrets["anthropic"]["api_key"]
        return anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize Anthropic client: {str(e)}")
        return None

def build_prompt(typed_text: str, founder_goal: str) -> str:
    """
    Modified prompt that emphasizes both the PDF content and founder goal
    """
    return f"""
You are an expert startup analyst. Analyze this startup comprehensively using the 10 Guard Rails framework, then generate targeted insights ONLY for genuine gaps.

--- MATERIALS ---
Founder Input: {typed_text}
CRITICAL - Assistance Goal: {founder_goal}
Pitch Deck: Please analyze the uploaded PDF document thoroughly for all business details, metrics, strategies, and technical specifications.

--- ABSOLUTE REQUIREMENTS ---
1. THOROUGHLY analyze the uploaded PDF pitch deck - extract ALL business information, metrics, strategies, market data, technical details, financial projections, etc.
2. MANDATORY: The founder specifically wants help with: "{founder_goal}" - This MUST be the primary focus of your analysis and recommendations
3. Every follow-up question and burning problem MUST directly address this assistance goal: "{founder_goal}"
4. If the founder goal is vague or unclear, ask specific clarifying questions about it
5. Your entire analysis should be shaped by understanding what specific help they're seeking

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
Before generating output, internally evaluate each guard rail based on BOTH the PDF content AND founder notes:

✅ WELL-COVERED = Specific details, metrics, clear strategy provided in PDF or founder notes
❌ WEAK/MISSING = Vague mentions, no specifics, or completely absent from both sources

**CRITICAL RULE: All follow-ups and burning problems must be filtered through the lens of their assistance goal: "{founder_goal}"**

--- OUTPUT FORMAT (JSON ONLY) ---
{{
  "title": "Product name with unique positioning from PDF/founder notes",
  "description": "3-4 sentences using actual metrics and specifics from PDF and founder input",
  "audience": "Precise roles/segments mentioned in PDF or founder notes",
  "problemStatements": ["3 specific pain points with business impact from analysis"],
  "tags": ["Industry/domain themes from PDF and founder content"],
  "followUpQuestions": ["Exactly 3 strategic questions directly addressing: {founder_goal}"],
  "burningProblems": ["Exactly 3 business risks from gaps, prioritized by relevance to: {founder_goal}"]
}}

--- FOLLOW-UP QUESTION REQUIREMENTS ---
Each question must be:
- **Directly related to their assistance goal**: "{founder_goal}"
- **2-3 sentences long** with analytical depth
- **Context-aware**: Reference specific details from PDF and founder notes
- **Strategic**: Address complex business challenges related to their requested help
- **Sophisticated**: Show deep understanding of how their situation relates to their assistance needs

EXAMPLE: If founder goal is "fundraising help", questions should focus on investor concerns, valuation, traction metrics, etc.
EXAMPLE: If founder goal is "product-market fit", questions should focus on customer validation, retention, usage patterns, etc.

CRITICAL SUCCESS CRITERIA:
- Anyone reading your output should immediately understand you've analyzed both the PDF content AND the founder's specific help request
- Your recommendations must be laser-focused on their assistance goal: "{founder_goal}"
- If you cannot determine how to help with their specific goal, ask clarifying questions about it

Return ONLY valid JSON, no additional text or formatting.
"""

def query_claude_with_pdf(client, prompt: str, pdf_content: bytes) -> str:
    """
    Send PDF directly to Claude for analysis using the correct document format
    """
    try:
        # Convert PDF to base64 for sending to Claude
        pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
        
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",  # Using Sonnet for PDF processing
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            max_tokens=2000,
        )
        
        return response.content[0].text if response.content else ""
        
    except Exception as e:
        raise Exception(f"Error processing with Claude: {str(e)}")

def extract_json(raw: str) -> t.Optional[dict]:
    """
    Robust JSON extraction from Claude response
    """
    cleaned = raw.strip()
    
    # Remove markdown code fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9]*\s*", "", cleaned)
        cleaned = re.sub(r"```$", "", cleaned).strip()
    
    # First attempt: direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: extract first JSON object
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None

def test_api_key(client):
    """Test if the Anthropic API key is working"""
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            messages=[{"role": "user", "content": "Hello, just testing the API key. Please respond with 'API working'."}],
            max_tokens=50
        )
        return True, response.content[0].text if response.content else "No response"
    except Exception as e:
        return False, str(e)

def main():
    st.title("🚀 Outlaw Idea Capture")
    st.markdown("### Analyze your startup using the 10 Guard Rails framework")
    
    # Initialize client
    client = get_anthropic_client()
    if not client:
        st.error("Cannot proceed without valid Anthropic API key")
        return
    
    # Sidebar for API key testing
    with st.sidebar:
        st.header("System Status")
        if st.button("Test API Key"):
            with st.spinner("Testing API connection..."):
                is_valid, message = test_api_key(client)
                if is_valid:
                    st.success(f"✅ API Key Valid: {message}")
                else:
                    st.error(f"❌ API Key Invalid: {message}")
        
        st.markdown("---")
        st.markdown("**Version:** 1.0.0")
        st.markdown("**Framework:** 10 Guard Rails")
    
    # Main form
    with st.form("startup_analysis_form"):
        st.subheader("📝 Founder Information")
        
        # Founder notes input
        founder_notes = st.text_area(
            "Founder Notes & Product Description",
            placeholder="Describe your startup, product, vision, and any key details...",
            height=150,
            help="Provide detailed information about your startup, product, and current status"
        )
        
        # Founder goal input
        founder_goal = st.text_area(
            "What specific help do you want from Outlaw?",
            placeholder="e.g., fundraising strategy, product-market fit validation, go-to-market planning...",
            height=100,
            help="Be specific about the type of assistance you're seeking"
        )
        
        st.subheader("📄 Pitch Deck Upload")
        
        # File upload
        pitch_deck = st.file_uploader(
            "Upload your pitch deck (PDF only)",
            type=['pdf'],
            help="Upload your startup pitch deck for comprehensive analysis"
        )
        
        # Submit button
        submitted = st.form_submit_button("🔍 Analyze Startup", use_container_width=True)
    
    # Process form submission
    if submitted:
        # Validation
        errors = []
        if not founder_notes.strip():
            errors.append("Founder notes cannot be empty")
        if not founder_goal.strip():
            errors.append("Founder goal cannot be empty")
        if not pitch_deck:
            errors.append("Please upload a PDF pitch deck")
        
        if errors:
            for error in errors:
                st.error(error)
            return
        
        # Process the analysis
        try:
            with st.spinner("🤖 Analyzing your startup with Claude AI..."):
                # Read PDF content
                pdf_content = pitch_deck.read()
                
                if len(pdf_content) == 0:
                    st.error("PDF file is empty")
                    return
                
                # Build prompt
                prompt = build_prompt(founder_notes, founder_goal)
                
                # Query Claude
                raw_output = query_claude_with_pdf(client, prompt, pdf_content)
                
                # Parse JSON response
                parsed_result = extract_json(raw_output)
                
                if parsed_result:
                    st.success("✅ Analysis completed successfully!")
                    display_results(parsed_result)
                else:
                    st.warning("⚠️ Could not parse structured output")
                    st.subheader("Raw Output")
                    st.text_area("Raw Claude Response", raw_output, height=300)
                    
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")

def display_results(data: dict):
    """Display the analysis results in a structured format"""
    
    # Title and Description
    st.header(f"📊 Analysis: {data.get('title', 'Unknown')}")
    st.markdown(f"**Description:** {data.get('description', 'No description available')}")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Audience
        st.subheader("🎯 Target Audience")
        st.write(data.get('audience', 'Not specified'))
        
        # Problem Statements
        st.subheader("❗ Problem Statements")
        problems = data.get('problemStatements', [])
        for i, problem in enumerate(problems, 1):
            st.write(f"{i}. {problem}")
        
        # Tags
        st.subheader("🏷️ Tags")
        tags = data.get('tags', [])
        if tags:
            st.write(" • ".join(tags))
        else:
            st.write("No tags available")
    
    with col2:
        # Follow-up Questions
        st.subheader("❓ Strategic Follow-up Questions")
        questions = data.get('followUpQuestions', [])
        for i, question in enumerate(questions, 1):
            st.write(f"{i}. {question}")
        
        # Burning Problems
        st.subheader("🔥 Critical Business Risks")
        burning_problems = data.get('burningProblems', [])
        for i, problem in enumerate(burning_problems, 1):
            st.write(f"{i}. {problem}")
    
    # Download results as JSON
    st.subheader("📥 Export Results")
    json_str = json.dumps(data, indent=2)
    st.download_button(
        label="Download Analysis as JSON",
        data=json_str,
        file_name=f"startup_analysis_{data.get('title', 'unknown').replace(' ', '_').lower()}.json",
        mime="application/json"
    )

if __name__ == "__main__":
    main()