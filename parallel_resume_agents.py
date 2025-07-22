import os
import asyncio
import time
from typing import List, Optional
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from langfuse import observe


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

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

class CompanyDetailsResponse(BaseModel):
    enriched_experience: List[EnrichedExperienceItem]

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

# Phase 1 Agents
@observe(name="personal_info_extractor")
async def personal_info_extractor(resume_text: str) -> PersonalInfo:
    """Extract personal information from resume"""
    start_time = time.time()
    print(f"⏱️  Personal Info Agent: Starting extraction...")
    
    prompt = """You are a personal information extraction specialist. Extract ONLY the following from the resume:
    1. Candidate's full name
    2. Email address (check personal email, work emails, LinkedIn profiles)
    3. Phone number
    4. Skills (list all technical skills based on resume, maximum 5 skills)
    5. Suggested Role based on experience and skills
    
    Focus ONLY on personal details. Do not extract company or education information.
    """
    
    completion = client.beta.chat.completions.parse(
        model="gpt-4.1-nano-2025-04-14",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": resume_text}
        ],
        response_format=PersonalInfoResponse,
    )
    
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    print(f"✅ Personal Info Agent: Completed in {duration}s")
    
    return completion.choices[0].message.parsed.personal_info

@observe(name="education_info_extractor")
async def education_info_extractor(resume_text: str) -> List[EducationItem]:
    """Extract education information from resume"""
    start_time = time.time()
    print(f"⏱️  Education Agent: Starting extraction...")
    
    prompt = """You are an education information extraction specialist. Extract ONLY education details:
    1. College/University name
    2. Course/degree
    3. Graduation year
    
    Focus ONLY on educational background. Do not extract personal or company information.
    """
    
    completion = client.beta.chat.completions.parse(
        model="gpt-4.1-nano-2025-04-14",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": resume_text}
        ],
        response_format=EducationInfoResponse,
    )
    
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    print(f"✅ Education Agent: Completed in {duration}s")
    
    return completion.choices[0].message.parsed.education

@observe(name="experience_info_extractor")
async def experience_info_extractor(resume_text: str) -> List[BasicExperienceItem]:
    """Extract basic experience information from resume"""
    start_time = time.time()
    print(f"⏱️  Experience Agent: Starting extraction...")
    
    prompt = """You are a work experience extraction specialist. Extract ONLY basic company experience:
    1. Company name (LOOK CAREFULLY - check email domains, LinkedIn URLs, official company names)
    2. Position/role
    3. Duration (EXACT start date and end date as they appear in resume)
    
    Focus ONLY on company name, position, and duration. Do not extract company details, funding, or other information.
    Extract dates EXACTLY as they appear in the resume without reformatting.
    """
    
    completion = client.beta.chat.completions.parse(
        model="gpt-4.1-nano-2025-04-14",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": resume_text}
        ],
        response_format=ExperienceInfoResponse,
    )
    
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    print(f"✅ Experience Agent: Completed in {duration}s")
    
    return completion.choices[0].message.parsed.experience

# Phase 2 Agents
@observe(name="stability_analyzer")
async def stability_analyzer(experience_list: List[BasicExperienceItem]) -> StabilityAnalysis:
    """Analyze stability and company matching from experience data"""
    start_time = time.time()
    print(f"⏱️  Stability Analyzer: Starting analysis...")
    
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
    experience_text = str(experience_data)
    
    completion = client.beta.chat.completions.parse(
        model="gpt-4.1-nano-2025-04-14",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Experience data: {experience_text}"}
        ],
        response_format=StabilityResponse,
    )
    
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    print(f"✅ Stability Analyzer: Completed in {duration}s")
    
    return completion.choices[0].message.parsed.stability_analysis

async def enrich_single_company(exp: BasicExperienceItem) -> EnrichedExperienceItem:
    """Enrich a single company's details with web search"""
    company_start_time = time.time()
    print(f"🔍 Enriching {exp.CompanyName}...")
    
    prompt = """You are a company details enrichment specialist. For the given company experience, provide:
    
    1. CompanyType: Product/Service/Banking (infer from company name)
    2. BusinessType: B2B/B2C/Banking (infer based on company type and nature)
    3. Location: Company main location
    4. NumberOfEmployees: if you can find specific data, otherwise null
    5. Funding: if you can find specific data, otherwise null
    
    COMPANY TYPE INFERENCE:
    - TCS, Infosys, Wipro, Accenture, Cognizant = Service companies (B2B)
    - Amazon, Google, Microsoft, Apple, Meta = Product companies
    - Banks = Banking companies
    
    BUSINESS TYPE INFERENCE:
    - Service companies = B2B
    - E-commerce (Amazon, Flipkart) = B2C
    - Enterprise software (Microsoft, Oracle) = B2B
    - Social media (Meta, Twitter) = B2C
    - Banking = Banking
    """
    
    exp_text = f"Company: {exp.CompanyName}, Position: {exp.Position}"
    
    completion = client.beta.chat.completions.parse(
        model= "gpt-4o-mini-search-preview",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": exp_text}
        ],
        response_format=CompanyDetailsResponse,
        web_search_options={},
    )
    
    # Create enriched item with basic details
    if completion.choices[0].message.parsed.enriched_experience:
        enriched_item = completion.choices[0].message.parsed.enriched_experience[0]
        enriched_item.CompanyName = exp.CompanyName
        enriched_item.Position = exp.Position
        enriched_item.Duration = exp.Duration
        
        company_end_time = time.time()
        company_duration = round(company_end_time - company_start_time, 2)
        print(f"✅ {exp.CompanyName} enriched in {company_duration}s")
        
        return enriched_item
    else:
        # Fallback if no response
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

@observe(name="company_details_enricher")
async def company_details_enricher(experience_list: List[BasicExperienceItem]) -> List[EnrichedExperienceItem]:
    """Enrich company details and perform web search for missing information - PARALLEL per company"""
    start_time = time.time()
    print(f"⏱️  Company Details Enricher: Starting parallel enrichment for {len(experience_list)} companies...")
    
    # Process each company in parallel
    tasks = [enrich_single_company(exp) for exp in experience_list]
    enriched_experience = await asyncio.gather(*tasks)
    
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    print(f"✅ Company Details Enricher: Completed in {duration}s (parallel per company)")
    
    return enriched_experience

# Orchestrator Functions
async def run_phase_1(resume_text: str) -> tuple[PersonalInfo, List[EducationItem], List[BasicExperienceItem]]:
    """Run Phase 1 agents in parallel"""
    phase_start_time = time.time()
    print("🚀 Starting Phase 1: Parallel extraction of personal info, education, and basic experience...")
    print("⏱️  All Phase 1 agents starting simultaneously...")
    
    # Track individual completion times
    completion_times = {}
    
    async def personal_info_with_timing():
        start = time.time()
        result = await personal_info_extractor(resume_text)
        completion_times['Personal Info'] = round(time.time() - start, 2)
        return result
    
    async def education_with_timing():
        start = time.time()
        result = await education_info_extractor(resume_text)
        completion_times['Education'] = round(time.time() - start, 2)
        return result
    
    async def experience_with_timing():
        start = time.time()
        result = await experience_info_extractor(resume_text)
        completion_times['Experience'] = round(time.time() - start, 2)
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
    
    print(f"✅ Phase 1 completed in {phase_duration}s (parallel execution)")
    print(f"📊 Phase 1 Agent Times: {completion_times}")
    print(f"🐌 Bottleneck: {bottleneck_agent} ({bottleneck_time}s)")
    
    return personal_info, education, experience

@observe(name="run_phase_2")
async def run_phase_2(experience_list: List[BasicExperienceItem]) -> tuple[StabilityAnalysis, List[EnrichedExperienceItem]]:
    """Run Phase 2 agents in parallel"""
    phase_start_time = time.time()
    print("🚀 Starting Phase 2: Parallel stability analysis and company details enrichment...")
    print("⏱️  All Phase 2 agents starting simultaneously...")
    
    # Track individual completion times
    completion_times = {}
    
    async def stability_with_timing():
        start = time.time()
        result = await stability_analyzer(experience_list)
        completion_times['Stability Analyzer'] = round(time.time() - start, 2)
        return result
    
    async def company_details_with_timing():
        start = time.time()
        result = await company_details_enricher(experience_list)
        completion_times['Company Details Enricher'] = round(time.time() - start, 2)
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
    
    print(f"✅ Phase 2 completed in {phase_duration}s (parallel execution)")
    print(f"📊 Phase 2 Agent Times: {completion_times}")
    print(f"🐌 Bottleneck: {bottleneck_agent} ({bottleneck_time}s)")
    
    return stability_analysis, enriched_experience

@observe(name="analyze_resume_parallel")
async def analyze_resume_parallel(resume_text: str) -> tuple[str, int]:
    """Main orchestrator function for parallel resume analysis"""
    total_start_time = time.time()
    print("🎯 Starting parallel resume analysis...")
    
    # Phase 1: Extract basic information in parallel
    personal_info, education, basic_experience = await run_phase_1(resume_text)
    
    # Phase 2: Analyze and enrich in parallel
    stability_analysis, enriched_experience = await run_phase_2(basic_experience)
    
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
    print(f"🎉 Parallel resume analysis completed successfully in {total_duration}s!")
    
    # Convert to JSON and return (simulate token count for consistency)
    json_output = combined_result.model_dump_json(indent=2)
    estimated_tokens = len(json_output.split()) * 2  # Rough estimate
    
    return json_output, estimated_tokens

# Convenience function for backward compatibility
async def analyze_resume(input_question: str) -> tuple[str, int]:
    """Backward compatible function that uses the new parallel approach"""
    return await analyze_resume_parallel(input_question)