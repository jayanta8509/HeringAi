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

class PositionItem(BaseModel):
    Position: str
    Duration: Duration

class BasicExperienceItem(BaseModel):
    CompanyName: str
    Positions: List[PositionItem]

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

class EnrichedPositionItem(BaseModel):
    Position: str
    Duration: Duration

class EnrichedExperienceItem(BaseModel):
    CompanyName: str
    Positions: List[EnrichedPositionItem]
    CompanyType: str
    BusinessType: str
    NumberOfEmployees: Optional[str] = None
    Funding: Optional[str] = None
    Location: str

# NEW: Batch Company Enrichment Response
class BatchCompanyEnrichmentResponse(BaseModel):
    enriched_companies: List[EnrichedExperienceItem]

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

# Phase 1 Agents (Same as before)
@observe(name="personal_info_extractor")
async def personal_info_extractor(resume_text: str) -> PersonalInfo:
    """Extract personal information from resume"""
    start_time = time.time()
    print(f"⏱️  Personal Info Agent: Starting extraction...")
    
    prompt = """Extract personal information from the resume text with high precision.

## Required Fields:
1. **CandidateFullName**: Extract the candidate's complete name (first, middle, last)
2. **EmailAddress**: Identify email address from contact info, LinkedIn profiles, or email signatures
3. **PhoneNumber**: Extract phone number in any format
4. **Skills**: List exactly 5 most relevant technical skills mentioned in resume
5. **SuggestedRole**: Recommend job role based on experience pattern and skills

## Instructions:
- Extract information exactly as written in resume
- For skills: prioritize programming languages, frameworks, and technical tools
- For role suggestion: match skills with experience level and domain
- Only extract personal contact information, ignore company/education details"""
    
    completion = client.beta.chat.completions.parse(
        model="gpt-4.1-nano-2025-04-14",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": resume_text}
        ],
        response_format=PersonalInfoResponse,
    )
    with open("resume_text.txt", "w") as f:
        f.write(resume_text)
    
    end_time = time.time()
    duration = round(end_time - start_time, 2)
    print(f"✅ Personal Info Agent: Completed in {duration}s")
    
    return completion.choices[0].message.parsed.personal_info

@observe(name="education_info_extractor")
async def education_info_extractor(resume_text: str) -> List[EducationItem]:
    """Extract education information from resume"""
    start_time = time.time()
    print(f"⏱️  Education Agent: Starting extraction...")
    
    prompt = """Extract educational background information from the resume.

## Required Fields:
1. **CollegeUniversity**: Full name of educational institution
2. **CourseDegree**: Complete degree name (e.g., "Bachelor of Technology in Computer Science")
3. **GraduationYear**: Year of graduation or expected graduation

## Instructions:
- Extract all educational entries (degrees, certifications, diplomas)
- Use exact institution names as written
- Include both completed and in-progress education
- If graduation year is not specified, use "N/A"
- Ignore company training or work-related education"""
    
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
    
    prompt = """
    Extract ALL work experience and group by company. Miss nothing.

## Primary Goal:
**Capture EVERY position mentioned** - full-time, part-time, internships, consulting, contract work, freelance, temporary roles.

## Extraction Rules:

### Complete Coverage:
- Scan entire resume systematically
- Check all sections: experience, summary, projects, anywhere employment is mentioned
- Include overlapping positions if held simultaneously
- Better to over-include than miss anything

### Grouping:
- **Same company = One entry** (promotions, role changes, rehires)
- **Different companies = Separate entries**
- Use exact company names as written

### Critical Points:
- Preserve original date formats exactly
- Use "Present"/"Current" as written for ongoing roles
- Include acting/interim/temporary positions
- Count total positions - does it match career length?

**Key**: If in doubt, include it. Complete capture is essential.
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
    print(completion.choices[0].message.parsed.experience)
    return completion.choices[0].message.parsed.experience

# Phase 2 Agents
@observe(name="stability_analyzer")
async def stability_analyzer(experience_list: List[BasicExperienceItem]) -> StabilityAnalysis:
    """Analyze stability and company matching from experience data"""
    start_time = time.time()
    print(f"⏱️  Stability Analyzer: Starting analysis...")
    
    prompt = """Analyze career stability and company experience patterns from the provided work history.

## Data Structure:
Each company has multiple positions with individual durations.

## Analysis Requirements:

### 1. StabilityAssessment (Array of strings)
- Calculate total tenure per company by summing all position durations at that company
- Format: "[CompanyName]: [X.XX] years"
- Round to 2 decimal places
- Combine all positions at same company for total tenure

### 2. AverageStability (Float)
- Calculate mean tenure across all companies (not positions)
- Use total company tenure from StabilityAssessment
- Exclude: internships, training programs, contract roles <6 months
- Return as float rounded to 2 decimals

### 3. Company Pattern Analysis
- **CompanyTypeMatch**: Categorize as Product/Service/Banking based on business model
- **BusinessTypeMatch**: Classify customer base as B2B/B2C/Banking

### 4. ComplexWorkExperience (Boolean)
- Return true if candidate has experience at companies with:
  - Large-scale product development
  - Advanced technology/engineering challenges
  - Significant market presence or innovation

## Company Classification Rules:
**Service**: TCS, Infosys, Wipro, Accenture, Cognizant, IBM, consulting firms
**Product**: Amazon, Google, Microsoft, Apple, Meta, Netflix, tech startups
**Banking**: Banks, fintech, financial services

## Calculation Method:
- For each company, sum durations of all positions held there
- Use exact dates provided to calculate tenure in years = (end_date - start_date) / 365.25
- Handle overlapping positions appropriately"""
    
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

# NEW: Batch Company Enrichment Agent
@observe(name="batch_company_enricher_openai")
async def batch_company_enricher_openai(experience_list: List[BasicExperienceItem]) -> List[EnrichedExperienceItem]:
    """Enrich ALL companies in a single batch request using OpenAI with web search"""
    start_time = time.time()
    print(f"⏱️  Batch Company Enricher (OpenAI): Starting enrichment for {len(experience_list)} companies...")
    
    # Create company list for the prompt
    companies_list = []
    for i, exp in enumerate(experience_list, 1):
        positions_text = []
        for pos in exp.Positions:
            positions_text.append(f"    - {pos.Position}: {pos.Duration.StartDate} to {pos.Duration.EndDate}")
        companies_list.append(f"{i}. {exp.CompanyName}\n" + "\n".join(positions_text))
    
    companies_text = "\n\n".join(companies_list)
    
    batch_prompt = f"""Enrich company information for all {len(experience_list)} companies listed below.

## Companies to Analyze:
{companies_text}

## Required Information for Each Company:
1. **CompanyType**: Product/Service/Banking
2. **BusinessType**: B2B/B2C/Banking 
3. **Location**: Primary headquarters city and country
4. **NumberOfEmployees**: Current headcount (null if unknown)
5. **Funding**: Funding status/valuation (null if unknown)

**Note**: Each company may have multiple positions listed. Enrich the company information once, but preserve all position details.

## Classification Framework:

### CompanyType Rules:
- **Product**: Tech companies building software/hardware products (Google, Apple, startups)
- **Service**: Consulting, outsourcing, professional services (TCS, Accenture, IBM)
- **Banking**: Financial institutions, fintech, investment firms

### BusinessType Rules:
- **B2B**: Sells to businesses (enterprise software, consulting)
- **B2C**: Sells to consumers (social media, e-commerce, retail)
- **Banking**: Financial services sector

### Reference Examples:
- Amazon, Google, Microsoft → Product companies
- TCS, Infosys, Wipro, Accenture → Service companies (B2B)
- JPMorgan, Goldman Sachs → Banking companies
- Facebook, Netflix → Product companies (B2C)

## Critical Requirements:
- Process companies in exact listed order
- Preserve original CompanyName and all position details unchanged
- Return structured data for all {len(experience_list)} companies
- Each company should have Positions array with all roles preserved
- Use web search for accurate, current information"""
    
    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini-search-preview",
            messages=[
                {"role": "system", "content": batch_prompt},
                {"role": "user", "content": f"Please enrich all {len(experience_list)} companies listed above"}
            ],
            response_format=BatchCompanyEnrichmentResponse,
            web_search_options={},
        )
        
        enriched_response = completion.choices[0].message.parsed
        
        # Validate we got the right number of companies
        if len(enriched_response.enriched_companies) != len(experience_list):
            print(f"⚠️  Warning: Expected {len(experience_list)} companies, got {len(enriched_response.enriched_companies)}")
        
        # Ensure core data are properly preserved from original
        final_enriched = []
        for i, original_exp in enumerate(experience_list):
            if i < len(enriched_response.enriched_companies):
                enriched_item = enriched_response.enriched_companies[i]
                # Force preserve the original data that should not change
                enriched_item.CompanyName = original_exp.CompanyName
                # Convert positions to EnrichedPositionItem
                enriched_positions = []
                for pos in original_exp.Positions:
                    enriched_positions.append(EnrichedPositionItem(
                        Position=pos.Position,
                        Duration=pos.Duration
                    ))
                enriched_item.Positions = enriched_positions
                final_enriched.append(enriched_item)
            else:
                # Fallback if we didn't get enough companies
                enriched_positions = []
                for pos in original_exp.Positions:
                    enriched_positions.append(EnrichedPositionItem(
                        Position=pos.Position,
                        Duration=pos.Duration
                    ))
                fallback_item = EnrichedExperienceItem(
                    CompanyName=original_exp.CompanyName,
                    Positions=enriched_positions,
                    CompanyType="Unknown",
                    BusinessType="Unknown",
                    Location="Unknown",
                    NumberOfEmployees=None,
                    Funding=None
                )
                final_enriched.append(fallback_item)
        
        end_time = time.time()
        duration = round(end_time - start_time, 2)
        print(f"✅ Batch Company Enricher (OpenAI): Completed {len(final_enriched)} companies in {duration}s")
        
        return final_enriched
        
    except Exception as e:
        print(f"⚠️  Error in batch company enrichment: {e}")
        # Fallback: return basic structure for all companies
        fallback_companies = []
        for exp in experience_list:
            enriched_positions = []
            for pos in exp.Positions:
                enriched_positions.append(EnrichedPositionItem(
                    Position=pos.Position,
                    Duration=pos.Duration
                ))
            fallback_item = EnrichedExperienceItem(
                CompanyName=exp.CompanyName,
                Positions=enriched_positions,
                CompanyType="Unknown",
                BusinessType="Unknown",
                Location="Unknown",
                NumberOfEmployees=None,
                Funding=None
            )
            fallback_companies.append(fallback_item)
        
        end_time = time.time()
        duration = round(end_time - start_time, 2)
        print(f"⚠️  Batch Company Enricher (OpenAI): Fallback completed in {duration}s")
        
        return fallback_companies

# Orchestrator Functions
async def run_phase_1_batch(resume_text: str) -> tuple[PersonalInfo, List[EducationItem], List[BasicExperienceItem]]:
    """Run Phase 1 agents in parallel"""
    phase_start_time = time.time()
    print("🚀 Starting Phase 1 (Batch): Parallel extraction of personal info, education, and basic experience...")
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
    
    print(f"✅ Phase 1 (Batch) completed in {phase_duration}s (parallel execution)")
    print(f"📊 Phase 1 Agent Times: {completion_times}")
    print(f"🐌 Bottleneck: {bottleneck_agent} ({bottleneck_time}s)")
    
    # Check if parallel execution is working correctly
    if abs(phase_duration - bottleneck_time) > 0.5:  # Allow 0.5s tolerance
        print(f"⚠️  WARNING: Parallel execution may not be optimal. Expected ~{bottleneck_time}s, got {phase_duration}s")
        print(f"   This could indicate sequential execution or API rate limiting.")
    else:
        print(f"✅ Parallel execution confirmed: Total time ({phase_duration}s) ≈ Bottleneck time ({bottleneck_time}s)")
    
    return personal_info, education, experience

@observe(name="run_phase_2_batch")
async def run_phase_2_batch(experience_list: List[BasicExperienceItem]) -> tuple[StabilityAnalysis, List[EnrichedExperienceItem]]:
    """Run Phase 2 agents in parallel - WITH ADAPTIVE COMPANY ENRICHMENT (Batch ≤3, Chunked >3)"""
    phase_start_time = time.time()
    print(f"🚀 Starting Phase 2: Parallel stability analysis and {'CHUNKED' if len(experience_list) > 3 else 'BATCH'} company enrichment...")
    print("⏱️  All Phase 2 agents starting simultaneously...")
    
    # Track individual completion times
    completion_times = {}
    
    async def stability_with_timing():
        start = time.time()
        result = await stability_analyzer(experience_list)
        completion_times['Stability Analyzer'] = round(time.time() - start, 2)
        return result
    
    async def batch_company_enrichment_with_timing():
        start = time.time()
        
        if len(experience_list) <= 3:
            # Single batch approach for 3 or fewer companies
            print(f"📦 Processing {len(experience_list)} companies in single batch")
            result = await batch_company_enricher_openai(experience_list)
            # Use total time for single batch
            completion_times['Batch Company Enricher'] = round(time.time() - start, 2)
        else:
            # Chunked parallel approach for more than 3 companies
            print(f"📦 Processing {len(experience_list)} companies in chunked parallel mode (3 per chunk)")
            
            # Split into chunks of 3
            chunks = []
            for i in range(0, len(experience_list), 3):
                chunk = experience_list[i:i+3]
                chunks.append(chunk)
            
            print(f"🔄 Created {len(chunks)} chunks: {[len(chunk) for chunk in chunks]}")
            
            # Process chunks in parallel
            chunk_tasks = []
            for i, chunk in enumerate(chunks):
                print(f"⏳ Chunk {i+1}: Processing {len(chunk)} companies")
                chunk_tasks.append(batch_company_enricher_openai(chunk))
            
            # Wait for all chunks to complete and track actual parallel time
            chunk_start_time = time.time()
            chunk_results = await asyncio.gather(*chunk_tasks)
            chunk_end_time = time.time()
            
            parallel_duration = round(chunk_end_time - chunk_start_time, 2)
            
            # Combine results from all chunks
            result = []
            for chunk_result in chunk_results:
                result.extend(chunk_result)
            
            print(f"✅ Combined results: {len(result)} companies total")
            print(f"⚡ Actual parallel execution time: {parallel_duration}s")
            
            # Use actual parallel time for bottleneck analysis in chunked mode
            completion_times['Batch Company Enricher'] = parallel_duration
            print(f"📊 Timing: Using parallel time ({parallel_duration}s) for bottleneck analysis")
        
        return result
    
    tasks = [
        stability_with_timing(),
        batch_company_enrichment_with_timing()
    ]
    
    stability_analysis, enriched_experience = await asyncio.gather(*tasks)
    
    phase_end_time = time.time()
    phase_duration = round(phase_end_time - phase_start_time, 2)
    
    # Find the bottleneck
    bottleneck_agent = max(completion_times, key=completion_times.get)
    bottleneck_time = completion_times[bottleneck_agent]
    
    enrichment_mode = 'Chunked' if len(experience_list) > 3 else 'Batch'
    print(f"✅ Phase 2 ({enrichment_mode}) completed in {phase_duration}s (parallel execution)")
    print(f"📊 Phase 2 Agent Times: {completion_times}")
    print(f"🐌 Bottleneck: {bottleneck_agent} ({bottleneck_time}s)")
    
    # Check if parallel execution is working correctly
    if abs(phase_duration - bottleneck_time) > 0.5:  # Allow 0.5s tolerance
        print(f"⚠️  WARNING: Parallel execution may not be optimal. Expected ~{bottleneck_time}s, got {phase_duration}s")
        print(f"   This could indicate sequential execution or API rate limiting.")
    else:
        print(f"✅ Parallel execution confirmed: Total time ({phase_duration}s) ≈ Bottleneck time ({bottleneck_time}s)")
    
    return stability_analysis, enriched_experience

@observe(name="analyze_resume_batch")
async def analyze_resume_batch(resume_text: str) -> tuple[str, int]:
    """Main orchestrator function for ADAPTIVE parallel resume analysis using OpenAI (Batch ≤3, Chunked >3)"""
    total_start_time = time.time()
    print("🎯 Starting ADAPTIVE parallel resume analysis with OpenAI...")
    
    # Phase 1: Extract basic information in parallel
    personal_info, education, basic_experience = await run_phase_1_batch(resume_text)
    
    # Phase 2: Analyze and enrich in parallel (WITH ADAPTIVE COMPANY ENRICHMENT)
    stability_analysis, enriched_experience = await run_phase_2_batch(basic_experience)
    
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
    print(f"🎉 ADAPTIVE parallel resume analysis with OpenAI completed successfully in {total_duration}s!")
    
    # Convert to JSON and return (simulate token count for consistency)
    json_output = combined_result.model_dump_json(indent=2)
    estimated_tokens = len(json_output.split()) * 2  # Rough estimate
    
    return json_output, estimated_tokens

# Convenience function for backward compatibility
async def analyze_resume(input_question: str) -> tuple[str, int]:
    """Backward compatible function that uses the new ADAPTIVE parallel approach (Batch ≤3, Chunked >3)"""
    return await analyze_resume_batch(input_question)

# Test function to verify parallel execution
async def test_parallel_execution():
    """Test function to verify that parallel execution is working correctly"""
    print("🧪 Testing parallel execution...")
    
    async def slow_task(name: str, delay: float):
        start = time.time()
        await asyncio.sleep(delay)  # Simulate work
        duration = round(time.time() - start, 2)
        print(f"   {name}: Completed in {duration}s")
        return duration
    
    # Test with 3 tasks: 2s, 3s, 1s
    start_time = time.time()
    results = await asyncio.gather(
        slow_task("Task A (2s)", 2),
        slow_task("Task B (3s)", 3), 
        slow_task("Task C (1s)", 1)
    )
    total_time = round(time.time() - start_time, 2)
    
    bottleneck_time = max(results)
    print(f"📊 Individual times: {results}")
    print(f"🐌 Bottleneck time: {bottleneck_time}s")
    print(f"⏱️  Total execution time: {total_time}s")
    
    if abs(total_time - bottleneck_time) < 0.1:
        print("✅ Parallel execution working correctly!")
    else:
        print("❌ Parallel execution not working - tasks may be running sequentially")
    
    return total_time, bottleneck_time

# Uncomment to run test
# if __name__ == "__main__":
#     asyncio.run(test_parallel_execution())