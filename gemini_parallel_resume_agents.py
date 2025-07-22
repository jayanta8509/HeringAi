import os
import asyncio
import time
import json
from typing import List, Optional, Any
from pydantic import BaseModel
from dotenv import load_dotenv
from langfuse import observe
from langchain_google_genai import ChatGoogleGenerativeAI
from google import genai
from google.genai import types

load_dotenv()

# Initialize Gemini client for web search
genai_client = genai.Client()

# Phase 1 Models
class PersonalInfo(BaseModel):
    CandidateFullName: str
    EmailAddress: str
    PhoneNumber: str
    Skills: List[str]
    SuggestedRole: str

class Duration(BaseModel):
    StartDate: str
    EndDate: str

class BasicExperienceItem(BaseModel):
    CompanyName: str
    Position: str
    Duration: Duration

class EducationItem(BaseModel):
    CollegeUniversity: str
    CourseDegree: str
    GraduationYear: str

class PersonalInfoResponse(BaseModel):
    personal_info: PersonalInfo

class ExperienceInfoResponse(BaseModel):
    experience: List[BasicExperienceItem]

class EducationInfoResponse(BaseModel):
    education: List[EducationItem]

# Phase 2 Models
class StabilityAnalysis(BaseModel):
    StabilityAssessment: List[str]
    AverageStability: str
    CompanyTypeMatch: str
    BusinessTypeMatch: str
    ComplexWorkExperience: bool

class EnrichedExperienceItem(BaseModel):
    CompanyName: str
    Position: str
    Duration: Duration
    CompanyType: str
    BusinessType: str
    NumberOfEmployees: Optional[str] = None
    Funding: Optional[str] = None
    Location: str

# Separate model for Gemini parsing that doesn't include Duration
class CompanyEnrichmentItem(BaseModel):
    CompanyType: str
    BusinessType: str
    NumberOfEmployees: Optional[str] = None
    Funding: Optional[str] = None
    Location: str

class CompanyDetailsResponse(BaseModel):
    company_details: CompanyEnrichmentItem

class StabilityResponse(BaseModel):
    stability_analysis: StabilityAnalysis

# Final Combined Model
class CombinedResumeData(BaseModel):
    CandidateFullName: str
    EmailAddress: str
    PhoneNumber: str
    Skills: List[str]
    SuggestedRole: str
    Experience: List[EnrichedExperienceItem]
    Education: List[EducationItem]
    StabilityAssessment: List[str]
    AverageStability: str
    CompanyTypeMatch: str
    BusinessTypeMatch: str
    ComplexWorkExperience: bool

# Utility Functions
def get_gemini_client(model: str = "gemini-2.5-flash-lite-preview-06-17", api_key: str = None) -> ChatGoogleGenerativeAI:
    """Initialize Gemini client with the specified model"""
    api_key = api_key or os.environ.get("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=0.1,
        google_api_key=api_key
    )

async def gemini_structured_completion(prompt: str, user_input: str, response_model: BaseModel, model: str = "gemini-2.5-flash-lite-preview-06-17") -> Any:
    """
    Generic function for Gemini structured completions with Pydantic models
    """
    llm = get_gemini_client(model)
    
    # Add JSON schema instructions to the prompt
    schema_str = response_model.model_json_schema()
    structured_prompt = f"""{prompt}

IMPORTANT: Return your response as a valid JSON object that matches this exact schema:
{json.dumps(schema_str, indent=2)}

Your response must be valid JSON that can be parsed directly. Do not include any text before or after the JSON object.
"""
    
    # Generate response
    response = llm.invoke(f"{structured_prompt}\n\nUser Input: {user_input}")
    
    # Clean the response content to handle markdown code blocks
    content = response.content.strip()
    
    # Remove markdown code blocks if present
    if content.startswith('```json'):
        content = content[7:]  # Remove ```json
    if content.startswith('```'):
        content = content[3:]   # Remove ```
    if content.endswith('```'):
        content = content[:-3]  # Remove closing ```
    
    content = content.strip()
    
    # Parse the response content as JSON and validate with Pydantic
    try:
        response_dict = json.loads(content)
        return response_model(**response_dict)
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error parsing Gemini response: {e}")
        print(f"Original response: {response.content}")
        print(f"Cleaned content: {content}")
        raise

# Phase 1 Agents - Gemini Version
@observe(name="personal_info_extractor_gemini")
async def personal_info_extractor_gemini(resume_text: str) -> PersonalInfo:
    """Extract personal information from resume using Gemini"""
    start_time = time.time()
    print(f"⏱️  Personal Info Agent (Gemini): Starting extraction...")
    
    prompt = """You are a personal information extraction specialist. Extract ONLY the following from the resume:
    1. Candidate's full name
    2. Email address (check personal email, work emails, LinkedIn profiles)
    3. Phone number
    4. Skills (list all technical skills based on resume, maximum 5 skills)
    5. Suggested Role based on experience and skills
    
    Focus ONLY on personal details. Do not extract company or education information.
    """
    
    response = await gemini_structured_completion(
        prompt=prompt,
        user_input=resume_text,
        response_model=PersonalInfoResponse,
        model="gemini-2.5-flash-lite-preview-06-17"
    )
    
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    print(f"✅ Personal Info Agent (Gemini): Completed in {duration}s")
    
    return response.personal_info

@observe(name="education_info_extractor_gemini")
async def education_info_extractor_gemini(resume_text: str) -> List[EducationItem]:
    """Extract education information from resume using Gemini"""
    start_time = time.time()
    print(f"⏱️  Education Agent (Gemini): Starting extraction...")
    
    prompt = """You are an education information extraction specialist. Extract ONLY education details:
    1. College/University name
    2. Course/degree
    3. Graduation year
    
    Focus ONLY on educational background. Do not extract personal or company information.
    """
    
    response = await gemini_structured_completion(
        prompt=prompt,
        user_input=resume_text,
        response_model=EducationInfoResponse,
        model="gemini-2.5-flash-lite-preview-06-17"
    )
    
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    print(f"✅ Education Agent (Gemini): Completed in {duration}s")
    
    return response.education

@observe(name="experience_info_extractor_gemini")
async def experience_info_extractor_gemini(resume_text: str) -> List[BasicExperienceItem]:
    """Extract basic experience information from resume using Gemini"""
    start_time = time.time()
    print(f"⏱️  Experience Agent (Gemini): Starting extraction...")
    
    prompt = """You are a work experience extraction specialist. Extract ONLY basic company experience:
    1. Company name (LOOK CAREFULLY - check email domains, LinkedIn URLs, official company names)
    2. Position/role
    3. Duration (EXACT start date and end date as they appear in resume)
    
    Focus ONLY on company name, position, and duration. Do not extract company details, funding, or other information.
    Extract dates EXACTLY as they appear in the resume without reformatting.
    """
    
    response = await gemini_structured_completion(
        prompt=prompt,
        user_input=resume_text,
        response_model=ExperienceInfoResponse,
        model="gemini-2.5-flash-lite-preview-06-17"
    )
    
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    print(f"✅ Experience Agent (Gemini): Completed in {duration}s")
    
    return response.experience

# Phase 2 Agents - Gemini Version
@observe(name="stability_analyzer_gemini")
async def stability_analyzer_gemini(experience_list: List[BasicExperienceItem]) -> StabilityAnalysis:
    """Analyze stability and company matching from experience data using Gemini"""
    start_time = time.time()
    print(f"⏱️  Stability Analyzer (Gemini): Starting analysis...")
    
    prompt = """You are a stability and company analysis specialist. Based on the experience data provided, calculate:
    
    1. Stability assessment (company-wise tenure duration as an array):
       - For each unique company, sum total tenure duration across all stints
       - Provide company name and total tenure in years (rounded to two decimal places)
       - Format: "Amazon: 1.16 years"
       - Output as array of strings, one per unique company
    
    2. Average Stability Across All Companies:
       - Calculate tenure in years for each full-time experience
       - Ignore internships or training roles
       - Compute average stability and return as float rounded to two decimal places
    
    3. Company type match and Business type match:
       - Infer company types (Product/Service/Banking) based on company names
       - Determine overall match pattern
    
    4. Complex Work Experience:
       - Evaluate if candidate worked at companies with complex product development
       - Consider ownership, scale, deep tech, product-led culture, reputation
       - Return true if at least one company meets these standards
    
    COMPANY TYPE INFERENCE RULES:
    - TCS, Infosys, Wipro, Accenture, Cognizant, IBM = Service companies
    - Amazon, Google, Microsoft, Apple, Meta, Netflix = Product companies
    - Banks, Consulting firms = Banking companies
    """
    
    experience_data = [exp.model_dump() for exp in experience_list]
    experience_text = json.dumps(experience_data, indent=2)
    
    response = await gemini_structured_completion(
        prompt=prompt,
        user_input=f"Experience data: {experience_text}",
        response_model=StabilityResponse,
        model="gemini-2.5-flash-lite-preview-06-17"
    )
    
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    print(f"✅ Stability Analyzer (Gemini): Completed in {duration}s")
    
    return response.stability_analysis

async def enrich_single_company_gemini_with_search(exp: BasicExperienceItem) -> EnrichedExperienceItem:
    """Enrich a single company's details using Gemini with Google web search"""
    company_start_time = time.time()
    print(f"🔍 Enriching {exp.CompanyName} (Gemini + Web Search)...")
    
    # Configure web search tool
    retrieval_tool = types.Tool(
        google_search_retrieval=types.GoogleSearchRetrieval(
            dynamic_retrieval_config=types.DynamicRetrievalConfig(
                mode=types.DynamicRetrievalConfigMode.MODE_DYNAMIC,
                dynamic_threshold=0.7  # Only search if confidence > 70%
            )
        )
    )
    
    config = types.GenerateContentConfig(
        tools=[retrieval_tool]
    )
    
    search_prompt = f"""
    I need detailed information about the company '{exp.CompanyName}'. Please provide:
    
    1. Company Type: Is it a Product, Service, or Banking company?
    2. Business Type: Is it B2B, B2C, or Banking focused?
    3. Main Location: Where is the company headquartered?
    4. Number of Employees: Current workforce size/headcount
    5. Funding Information: Total funding raised, valuation, or if it's public
    
    For context, this person worked as a {exp.Position} at {exp.CompanyName}.
    
    Please search for current, accurate information about this company and provide specific details where available.
    """
    
    try:
        # Use Gemini with web search
        response = genai_client.models.generate_content(
            model='gemini-1.5-flash',
            contents=search_prompt,
            config=config,
            response_model=CompanyDetailsResponse
        )
        
        search_results = response.text
        grounded = response.candidates[0].grounding_metadata is not None
        
        if grounded:
            print(f"🌐 {exp.CompanyName}: Web search provided additional data")
        else:
            print(f"🧠 {exp.CompanyName}: Using model knowledge only")
        
        # Now use structured completion to parse the enriched data
        parsing_prompt = f"""Based on the company information provided, extract structured data for {exp.CompanyName}:

Company Information:
{search_results}

Extract the following in the exact JSON format requested:
1. CompanyType: Product/Service/Banking (REQUIRED - never use null)
2. BusinessType: B2B/B2C/Banking (REQUIRED - never use null)
3. Location: Main company location (REQUIRED - never use null)
4. NumberOfEmployees: Specific count if available, otherwise "Unknown"
5. Funding: Funding amount/status if available, otherwise "Unknown"

IMPORTANT: 
- Never return null for CompanyType, BusinessType, or Location
- Use "Unknown" instead of null for missing data
- For Duration, DO NOT include StartDate or EndDate - they will be set separately

INFERENCE RULES:
- TCS, Infosys, Wipro, Accenture, Cognizant = Service companies (B2B)
- Amazon, Google, Microsoft, Apple, Meta = Product companies
- Banks = Banking companies
- E-commerce platforms = B2C
- Enterprise software = B2B
- Social media = B2C
"""
        
        structured_response = await gemini_structured_completion(
            prompt=parsing_prompt,
            user_input=f"Company: {exp.CompanyName}, Position: {exp.Position}",
            response_model=CompanyDetailsResponse,
            model="gemini-2.0-flash"
        )
        
        # Create enriched item using the parsed company details
        if structured_response.company_details:
            company_details = structured_response.company_details
            enriched_item = EnrichedExperienceItem(
                CompanyName=exp.CompanyName,
                Position=exp.Position,
                Duration=exp.Duration,
                CompanyType=company_details.CompanyType,
                BusinessType=company_details.BusinessType,
                NumberOfEmployees=company_details.NumberOfEmployees,
                Funding=company_details.Funding,
                Location=company_details.Location
            )
            
            company_end_time = time.time()
            company_duration = round(company_end_time - company_start_time, 2)
            print(f"✅ {exp.CompanyName} enriched in {company_duration}s (Gemini + Search)")
            
            return enriched_item
        else:
            raise ValueError("No company details returned")
            
    except Exception as e:
        print(f"⚠️  Error enriching {exp.CompanyName}: {e}")
        # Fallback enrichment without web search
        return await enrich_single_company_gemini_fallback(exp)

async def enrich_single_company_gemini_fallback(exp: BasicExperienceItem) -> EnrichedExperienceItem:
    """Fallback enrichment without web search"""
    print(f"🔄 Using fallback enrichment for {exp.CompanyName}")
    
    prompt = """You are a company details enrichment specialist. For the given company experience, provide:
    
    1. CompanyType: Product/Service/Banking (infer from company name)
    2. BusinessType: B2B/B2C/Banking (infer based on company type and nature)
    3. Location: Company main location (use your knowledge)
    4. NumberOfEmployees: if you know from your training data, otherwise null
    5. Funding: if you know from your training data, otherwise null
    
    COMPANY TYPE INFERENCE:
    - TCS, Infosys, Wipro, Accenture, Cognizant = Service companies (B2B)
    - Amazon, Google, Microsoft, Apple, Meta = Product companies
    - Banks = Banking companies
    
    Use your existing knowledge to provide the best estimates possible.
    """
    
    try:
        response = await gemini_structured_completion(
            prompt=prompt,
            user_input=f"Company: {exp.CompanyName}, Position: {exp.Position}",
            response_model=CompanyDetailsResponse,
            model="gemini-2.5-flash-lite-preview-06-17"
        )
        
        if response.company_details:
            company_details = response.company_details
            enriched_item = EnrichedExperienceItem(
                CompanyName=exp.CompanyName,
                Position=exp.Position,
                Duration=exp.Duration,
                CompanyType=company_details.CompanyType,
                BusinessType=company_details.BusinessType,
                NumberOfEmployees=company_details.NumberOfEmployees,
                Funding=company_details.Funding,
                Location=company_details.Location
            )
            return enriched_item
        else:
            raise ValueError("No response from fallback")
            
    except Exception as e:
        print(f"⚠️  Fallback also failed for {exp.CompanyName}: {e}")
        # Final fallback with minimal data
        return EnrichedExperienceItem(
            CompanyName=exp.CompanyName,
            Position=exp.Position,
            Duration=exp.Duration,
            CompanyType="Unknown",
            BusinessType="Unknown",
            Location="Unknown",
            NumberOfEmployees=None,
            Funding=None
        )

@observe(name="company_details_enricher_gemini")
async def company_details_enricher_gemini(experience_list: List[BasicExperienceItem]) -> List[EnrichedExperienceItem]:
    """Enrich company details using Gemini with web search - PARALLEL per company"""
    start_time = time.time()
    print(f"⏱️  Company Details Enricher (Gemini + Web Search): Starting parallel enrichment for {len(experience_list)} companies...")
    
    # Process each company in parallel with web search
    tasks = [enrich_single_company_gemini_with_search(exp) for exp in experience_list]
    enriched_experience = await asyncio.gather(*tasks)
    
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    print(f"✅ Company Details Enricher (Gemini + Web Search): Completed in {duration}s (parallel per company)")
    
    return enriched_experience

# Orchestrator Functions - Gemini Version
async def run_phase_1_gemini(resume_text: str) -> tuple[PersonalInfo, List[EducationItem], List[BasicExperienceItem]]:
    """Run Phase 1 agents in parallel using Gemini"""
    phase_start_time = time.time()
    print("🚀 Starting Phase 1 (Gemini): Parallel extraction of personal info, education, and basic experience...")
    print("⏱️  All Phase 1 Gemini agents starting simultaneously...")
    
    # Track individual completion times
    completion_times = {}
    
    async def personal_info_with_timing():
        start = time.time()
        result = await personal_info_extractor_gemini(resume_text)
        completion_times['Personal Info (Gemini)'] = round(time.time() - start, 2)
        return result
    
    async def education_with_timing():
        start = time.time()
        result = await education_info_extractor_gemini(resume_text)
        completion_times['Education (Gemini)'] = round(time.time() - start, 2)
        return result
    
    async def experience_with_timing():
        start = time.time()
        result = await experience_info_extractor_gemini(resume_text)
        completion_times['Experience (Gemini)'] = round(time.time() - start, 2)
        return result
    
    tasks = [
        personal_info_with_timing(),
        education_with_timing(),
        experience_with_timing()
    ]
    
    personal_info, education, experience = await asyncio.gather(*tasks)
    
    phase_end_time = time.time()
    phase_duration = round(phase_end_time - phase_start_time, 2)
    
    # Find the bottleneck
    bottleneck_agent = max(completion_times, key=completion_times.get)
    bottleneck_time = completion_times[bottleneck_agent]
    
    print(f"✅ Phase 1 (Gemini) completed in {phase_duration}s (parallel execution)")
    print(f"📊 Phase 1 Gemini Agent Times: {completion_times}")
    print(f"🐌 Bottleneck: {bottleneck_agent} ({bottleneck_time}s)")
    
    return personal_info, education, experience

@observe(name="run_phase_2_gemini")
async def run_phase_2_gemini(experience_list: List[BasicExperienceItem]) -> tuple[StabilityAnalysis, List[EnrichedExperienceItem]]:
    """Run Phase 2 agents in parallel using Gemini"""
    phase_start_time = time.time()
    print("🚀 Starting Phase 2 (Gemini): Parallel stability analysis and company details enrichment...")
    print("⏱️  All Phase 2 Gemini agents starting simultaneously...")
    
    # Track individual completion times
    completion_times = {}
    
    async def stability_with_timing():
        start = time.time()
        result = await stability_analyzer_gemini(experience_list)
        completion_times['Stability Analyzer (Gemini)'] = round(time.time() - start, 2)
        return result
    
    async def company_details_with_timing():
        start = time.time()
        result = await company_details_enricher_gemini(experience_list)
        completion_times['Company Details Enricher (Gemini)'] = round(time.time() - start, 2)
        return result
    
    tasks = [
        stability_with_timing(),
        company_details_with_timing()
    ]
    
    stability_analysis, enriched_experience = await asyncio.gather(*tasks)
    
    phase_end_time = time.time()
    phase_duration = round(phase_end_time - phase_start_time, 2)
    
    # Find the bottleneck
    bottleneck_agent = max(completion_times, key=completion_times.get)
    bottleneck_time = completion_times[bottleneck_agent]
    
    print(f"✅ Phase 2 (Gemini) completed in {phase_duration}s (parallel execution)")
    print(f"📊 Phase 2 Gemini Agent Times: {completion_times}")
    print(f"🐌 Bottleneck: {bottleneck_agent} ({bottleneck_time}s)")
    
    return stability_analysis, enriched_experience

@observe(name="analyze_resume_parallel_gemini")
async def analyze_resume_parallel_gemini(resume_text: str) -> tuple[str, int]:
    """Main orchestrator function for parallel resume analysis using Gemini"""
    total_start_time = time.time()
    print("🎯 Starting parallel resume analysis with Gemini...")
    
    # Phase 1: Extract basic information in parallel
    personal_info, education, basic_experience = await run_phase_1_gemini(resume_text)
    
    # Phase 2: Analyze and enrich in parallel
    stability_analysis, enriched_experience = await run_phase_2_gemini(basic_experience)
    
    # Combine all results
    combined_result = CombinedResumeData(
        CandidateFullName=personal_info.CandidateFullName,
        EmailAddress=personal_info.EmailAddress,
        PhoneNumber=personal_info.PhoneNumber,
        Skills=personal_info.Skills,
        SuggestedRole=personal_info.SuggestedRole,
        Experience=enriched_experience,
        Education=education,
        StabilityAssessment=stability_analysis.StabilityAssessment,
        AverageStability=stability_analysis.AverageStability,
        CompanyTypeMatch=stability_analysis.CompanyTypeMatch,
        BusinessTypeMatch=stability_analysis.BusinessTypeMatch,
        ComplexWorkExperience=stability_analysis.ComplexWorkExperience
    )
    
    total_end_time = time.time()
    total_duration = round(total_end_time - total_start_time, 2)
    print(f"🎉 Parallel resume analysis with Gemini completed successfully in {total_duration}s!")
    
    # Convert to JSON and return (simulate token count for consistency)
    json_output = combined_result.model_dump_json(indent=2)
    estimated_tokens = len(json_output.split()) * 2  # Rough estimate
    
    return json_output, estimated_tokens

# Convenience function for backward compatibility
async def analyze_resume(input_question: str) -> tuple[str, int]:
    """Backward compatible function that uses the new parallel Gemini approach"""
    return await analyze_resume_parallel_gemini(input_question)