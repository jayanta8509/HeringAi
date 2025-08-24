import os
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from typing import List
import json
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
from langfuse import observe

class CompanyAnalysisItem(BaseModel):
    CompanyName: str
    CompanyType: str  # Product/Service/Banking
    IndustrySector: str
    BusinessModel: str  # B2B/B2C
    NotableAchievements: str

class EducationAssessment(BaseModel):
    UniversityAssessment: str
    CourseRelevance: str

class Step(BaseModel):
    AIRating: int  # 0-10 scale (converted from 100-point system)
    ShouldBeShortlisted: str  # Yes/No
    CompanyAnalysis: List[CompanyAnalysisItem]
    EducationAssessment: EducationAssessment
    MissingExpectations: List[str]
    OverallRecommendation: str  # 3-4 lines with mandatory experience breakdown
    AIShortlisted: str  # Yes/No
    InternalShortlisted: str  # Yes/No
    InterviewInProcess: str  # Yes/No
    FinalResult: bool  # true for Selected, false for Rejected
    CandidateJoined: str  # High/Medium/Low (likelihood of joining if offered)

class resume_data(BaseModel):
    steps: list[Step]

@observe(name="analyze_resume_and_jd")
def analyze_resume_and_jd(combined_input):
    prompt_template = """You are an expert recruitment assistant. Analyze how well the candidate matches the job description with comprehensive evaluation.

## Analysis Framework:

### 1. AI Rating Calculation (0-10 scale)
Calculate total points (0-100), then convert using conversion table below.

**IMPORTANT: The final AIRating field MUST be between 0-10, NOT the raw score!**

#### Scoring Components:
**a) Company Type Match - Tech Perspective (25 points max):**
- 25 points: Perfect match (Product-based tech candidate + Product-based tech company JD, OR Service-based tech candidate + Service-based tech company JD)
- 18 points: Mixed tech experience (both Product + Service-based tech) matching either type of hiring company
- 10 points: Partial mismatch (Product-based tech experience + Service-based tech company JD, OR Service-based tech experience + Product-based tech company JD)
- 0 points: Complete mismatch or unclear tech company types

**Company Type Definitions:**
- **Product-Based Tech Companies**: Build and maintain their own software products, platforms, or SaaS solutions (e.g., Google, Microsoft, Netflix, Spotify, Airbnb, internal product teams)
- **Service-Based Tech Companies**: Provide technology consulting, development services, or solutions for clients (e.g., TCS, Infosys, Wipro, Accenture, consulting firms, agencies)
- **Banking/Financial Tech**: Traditional banks or fintech companies with internal tech teams
- **Hybrid Companies**: Mix of product and service models

**b) Domain Match - Business Type (15 points max):**
- 15 points: Exact match (B2B to B2B, B2C to B2C)
- 10 points: Compatible mixed models (B2C/B2B experience for B2B role)
- 6 points: Somewhat relevant
- 0 points: No relevance

**c) Keywords + Skill Set Match (20 points max):**
- 20 points: 80%+ skills match JD tech stack
- 16 points: 60-79% skills match
- 12 points: 40-59% skills match
- 8 points: 20-39% skills match
- 4 points: <20% skills match
- 0 points: No relevant skills

**d) Role Match (10 points max):**
- 10 points: Exact role match (Frontend to Frontend, Backend to Backend)
- 7 points: Related role (Full-stack to Frontend/Backend)
- 3 points: Transferable role
- 0 points: Fundamental mismatch

**e) Responsibilities Match (15 points max):**
- 15 points: Candidate's projects/work clearly maps to JD expectations
- 10 points: Partially relevant experience
- 5 points: Loosely related
- 0 points: No relevant project experience

**f) College Prestige (3 points max):**
- 3 points: Top-tier (IIT, NIT, MIT, BITS Stanford, etc.)
- 2 points: Good tier (State universities, reputable colleges)
- 1 point: Average tier
- 0 points: Unknown/lesser-known institutions

**g) Awards & Recognition (3 points max):**
- 3 points: Relevant certifications, awards, recognitions related to JD
- 1-2 points: Some certifications/awards
- 0 points: None mentioned

**h) Experience Level Match (5 points max):**
- 5 points: Experience level perfectly matches JD requirements (junior to junior, senior to senior)
- 3 points: Slightly over/under qualified but manageable
- 1 point: Significantly over/under qualified
- 0 points: Completely mismatched experience level

**i) Work Experience Distribution Analysis (4 points max):**
- 4 points: Experience distribution perfectly aligns with hiring company type (70%+ matching experience)
- 3 points: Good alignment with 50-69% matching experience
- 2 points: Moderate alignment with 30-49% matching experience  
- 1 point: Limited alignment with 10-29% matching experience
- 0 points: Poor alignment with <10% matching experience or unclear work evidence

**j) Resume Quality & Presentation (2 points max):**
- 2 points: Well-structured, professional resume with clear achievements and quantified results
- 1 point: Good resume structure with basic achievements
- 0 points: Poor resume quality or missing critical information

#### MANDATORY CONVERSION TABLE - USE THIS TO CONVERT TO FINAL AIRating:
- 90-100 points = AI Rating 9 or 10
- 80-89 points = AI Rating 8
- 70-79 points = AI Rating 7
- 60-69 points = AI Rating 6
- 50-59 points = AI Rating 5
- 40-49 points = AI Rating 4
- 30-39 points = AI Rating 3
- 20-29 points = AI Rating 2
- 10-19 points = AI Rating 1
- 0-9 points = AI Rating 0

**EXAMPLE: If you calculate 21 total points, the AIRating field should be 2, NOT 21!**

### 2. Shortlisting Decision
Whether the candidate should be shortlisted (Yes/No):
- Should be "No" if there's a fundamental role mismatch (backend candidate for frontend role or vice versa)
- Rating should be ≤4 if there's a fundamental role mismatch (backend candidate for frontend job)
- Rating should consider skill relevance percentage: if <50% skills match, rating should be ≤5

### 3. Company Analysis Requirements
Analysis of each company in the candidate's resume:
- **Company name**
- **Company type from tech perspective**: 
  - **Product-based**: Companies building their own tech products/platforms
  - **Service-based**: Companies providing tech consulting/development services to clients
  - **Banking/Financial**: Traditional banks or fintech with internal tech teams
  - **Hybrid**: Mix of product development and client services
- **Industry sector** (e.g., E-commerce, Healthcare, Fintech, Media, etc.)
- **Business model** (B2B/B2C/B2B2C)
- **Notable achievements** (specific to candidate's work at that company)

### 4. Education Assessment
- College/University assessment
- Course relevance

### 5. Gap Analysis - Anything Missing as per Expectations in the JD
Identify missing requirements:
- Include fundamental role mismatches
- Include responsibilities mismatches
- Highlight missing core skills for the specific role
- Note experience level gaps

### 6. Overall Recommendation Format (Detailed Summary in 3-4 lines)
**Requirements:**
- **CRITICAL: Write in perfect, grammatically correct English with proper sentence structure.**
- **Use professional, clear, and concise language without any grammatical errors.**
- **MUST include work experience breakdown by company type as percentages.**
- Calculate total years of experience and show percentage split: "X% service-based tech companies, Y% product-based tech companies, and Z% banking/financial tech companies based on work experience."
- For product-based tech companies, further divide experience into business models:
  - Calculate the percentage of B2B vs B2C experience **based on the duration spent at each product-based tech company.**
  - If the product-based tech experience includes only B2B or B2C, mention that.
  - If it includes both, state the exact percentage like: "Product-based tech company work includes 63% B2B and 37% B2C."
- For service-based tech companies, business type is always considered as client services.
- For banking/financial companies, business type is always considered Banking/Financial tech.
- **Tech Company Type Clarification**: Ensure proper classification of each company as product-focused vs service-focused from technology perspective.
- Ensure smooth flow, accurate calculations, proper punctuation, and subject-verb agreement.
- **Final Evaluation (based on FinalResult boolean):**
  - If FinalResult = true, add: "This candidate is suitable for this role."
  - If FinalResult = false, add: "This candidate is not suitable for this role."

**Example:** "The candidate has 6 years of total experience: 33% in service-based tech companies, 50% in product-based tech companies, and 17% in banking/financial tech companies. The product-based tech experience includes 34% B2B and 66% B2C, offering well-rounded exposure across both segments. The service-based tech experience contributes solid client delivery and diverse technology exposure, while the banking tech experience adds domain-specific expertise. This candidate is suitable for this role."

### 7. Candidate Status Prediction
- Should be AI shortlisted (Yes/No)
- Should be internally shortlisted (Yes/No)
- Ready for interview process (Yes/No)
- **Final result prediction (Selected/Rejected)**
  - Based on the candidate's experience, skills, education, suggested role, total years of experience, and alignment with the job description (JD) requirements, determine whether the candidate is a good fit for the role.
  - "Selected" If the candidate is a good fit, return true.
  - "Rejected" If the candidate is not a good fit, return false.
- **Likelihood of joining if offered (High/Medium/Low)**

### Status Prediction Guidelines:
- **AIShortlisted should be "Yes" if the AIRating is 7 or higher, otherwise "No"**
- **InternalShortlisted should be your recommendation based on the candidate's fit**
- **InterviewInProcess should be "Yes" if you recommend they proceed to interviews**
- **FinalResult should be true if they're an excellent match, false if poor match**
- **CandidateJoined should be your prediction of whether they'd join if offered**

### 8. Work Experience Distribution Analysis
**MANDATORY: Calculate and analyze experience ratios based on duration at each company**

**Experience Ratio Calculation Requirements:**
- **Total Experience Breakdown**: Calculate exact percentages for each tech company type
  - **Product-based tech company experience**: X% (companies building own tech products)
  - **Service-based tech company experience**: Y% (companies providing tech services to clients)  
  - **Banking/Financial tech company experience**: Z% (banks/fintech with internal tech teams)
  - **Hybrid tech company experience**: W% (if applicable - mix of product + service)
- **Product Company Business Model Split**: Further break down product-based tech experience
  - B2B product experience: X% of total product experience
  - B2C product experience: Y% of total product experience
  - B2B2C/Mixed model experience: Z% of total product experience

**Work Evidence Validation:**
- **Project Descriptions**: Analyze depth and relevance of project descriptions
- **Technology Implementation**: Evidence of hands-on work with claimed technologies
- **Achievement Metrics**: Quantified results and measurable impact statements
- **Role Progression**: Evidence of growing responsibilities and skill development
- **Technical Contributions**: Specific examples of technical work and problem-solving

**Company Type Impact Assessment:**
- **Technical Skill Set Alignment**: How each tech company type experience contributes to role requirements
- **Development Process Familiarity**: Understanding of different methodologies (Product development cycles vs Client project cycles)
- **Scale and Architecture Experience**: Experience with different system scales (internal products vs client solutions)
- **Technology Stack Depth**: Deep product-focused tech skills vs diverse client-focused tech exposure
- **Innovation vs Delivery Focus**: Product innovation experience vs service delivery excellence
- **Ownership vs Service Mindset**: Long-term product ownership vs client satisfaction delivery

**Experience Quality vs Quantity Analysis:**
- **Depth Assessment**: Deep expertise in specific areas vs broad generalist experience  
- **Recency Factor**: How recent is the most relevant experience
- **Progression Trajectory**: Career growth pattern and learning curve evidence
- **Consistency Check**: Alignment between claimed skills and work history evidence
- **Experience Quality Assessment**: Evaluate depth vs breadth of experience
- **Career Progression Analysis**: Assess growth trajectory and role progression
- **Technology Stack Evolution**: Consider how candidate's tech stack has evolved
- **Project Complexity Evaluation**: Analyze complexity and scale of projects handled
- **Leadership and Mentoring**: Evaluate any leadership or mentoring experience
- **Cultural Fit Indicators**: Assess potential cultural alignment based on background
- **Market Demand Analysis**: Consider candidate's market value and demand
- **Retention Risk Assessment**: Evaluate likelihood of long-term retention

### 9. Advanced Analysis Considerations
- **Current Salary Evaluation**: Assess if candidate's current/expected salary aligns with budget
- **Salary Growth Trajectory**: Analyze salary progression and expectations
- **Total Compensation Package**: Consider benefits, equity, bonuses beyond base salary
- **Market Rate Comparison**: Compare candidate expectations with industry standards
- **Negotiation Flexibility**: Assess potential for salary negotiation
- **Cost-Benefit Analysis**: Evaluate ROI of hiring this candidate at expected compensation

### 10. Salary and Compensation Analysis
**Red Flags to Identify:**
- **Job Hopping Pattern**: >3 jobs in 2 years without valid reasons
- **Career Stagnation**: Same role/salary for >3 years without progression
- **Skill Gap Penalties**: Major technology gaps that require extensive training
- **Overqualification Risk**: Candidate may leave quickly for better opportunities
- **Notice Period Issues**: Extended notice periods that delay joining
- **Background Verification Risks**: Inconsistencies in resume vs claimed experience
- **Team Fit Concerns**: Previous feedback about collaboration or attitude issues


### 12. Competitive Analysis
**Technical Proficiency Indicators:**
- **Skill Recency**: How recently candidate has used required technologies
- **Technology Depth**: Evidence of deep expertise vs surface-level knowledge
- **Project Complexity**: Scale and complexity of projects mentioned in resume
- **Achievement Quantification**: Presence of metrics, numbers, and measurable outcomes
- **Continuous Learning**: Evidence of ongoing skill development and certifications

**Resume Content Analysis:**
- **Professional Presentation**: Layout, formatting, and overall visual appeal
- **Content Clarity**: Clear job descriptions, responsibilities, and achievements
- **Relevance Focus**: How well resume is tailored to the specific job role
- **Achievement Orientation**: Focus on accomplishments rather than just duties
- **Technical Keyword Density**: Appropriate use of relevant technical terminology
- **Career Narrative**: Logical career progression and story flow
- **Quantified Impact**: Specific metrics, percentages, or numbers showing results

### 13. Resume-Based Assessment Framework
**Priority Areas for Phone/Video Screening:**
- **Technical Stack Validation**: Verify depth of claimed technical skills
- **Project Details Deep-dive**: Get specifics about mentioned projects and achievements
- **Role Clarification**: Understand exact responsibilities in previous roles
- **Availability and Timeline**: Confirm notice period and joining timeline
- **Salary Expectations**: Initial discussion about compensation expectations



### 14. Executive Summary Template
**Quantitative Analysis:**
- **Skills Coverage Score**: X out of Y required skills (percentage)
- **Experience Relevance Ratio**: Relevant years / Total years
- **Technology Recency Score**: How current are the candidate's tech skills
- **Project Scale Indicator**: Average team size, budget, or user base handled
- **Career Velocity**: Role progression rate (promotions per year)

**Benchmarking Metrics:**
- **Peer Comparison**: How candidate ranks vs others interviewed for similar roles
- **Market Positioning**: Candidate's profile strength in current job market
- **Hiring Success Probability**: Likelihood of successful hire based on profile
- **Long-term Retention Score**: Probability of 2+ year retention

### 15. Data-Driven Insights
**Hiring Decision Framework:**
```
IF AIRating >= 8 AND No Red Flags → STRONG HIRE
IF AIRating 6-7 AND Manageable Gaps → CONDITIONAL HIRE  
IF AIRating 4-5 AND Major Concerns → RISKY HIRE
IF AIRating <= 3 → DO NOT HIRE
```

**Tie-Breaker Criteria** (when multiple candidates have similar ratings):
1. Cultural fit and team dynamics potential
2. Learning agility and growth mindset indicators  
3. Unique skills or experiences that add strategic value
4. Availability and joining timeline
5. Salary expectations vs budget alignment
6. Long-term career alignment with company growth

### 16. Decision Support Matrix
**COMPREHENSIVE VERIFICATION CHECKLIST:**
- [ ] AIRating converted correctly from 100-point to 0-10 scale  
- [ ] All 10 scoring components calculated and justified (including Work Experience Distribution)
- [ ] Experience percentages total 100% across all company types
- [ ] B2B/B2C breakdown provided for product company experience with specific percentages
- [ ] Work evidence validation completed for each claimed skill/technology
- [ ] Experience ratio impact on role suitability clearly analyzed
- [ ] Project descriptions depth assessed for technical credibility
- [ ] Risk assessment completed with specific red flags identified
- [ ] Executive summary prepared for management review
- [ ] Quantitative metrics calculated where possible
- [ ] Decision rationale clearly articulated
"""

    completion = client.beta.chat.completions.parse(
    model="gpt-4.1-nano-2025-04-14",
    messages=[
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": combined_input}
    ],
    response_format=resume_data,
    )

    math_reasoning = completion.choices[0].message

    # prompt_tokens = completion.usage.prompt_tokens
    # completion_tokens = completion.usage.completion_tokens
    total_tokens = completion.usage.total_tokens

    # If the model refuses to respond, you will get a refusal message
    if hasattr(math_reasoning, 'refusal') and math_reasoning.refusal:
        # print(math_reasoning.refusal)
        pass
    else:
        # Convert the parsed response to a Pydantic model
        math_solution = resume_data(steps=math_reasoning.parsed.steps)
        
        # Fix AI Rating if it's > 10 (convert from 100-point scale)
        for step in math_solution.steps:
            if step.AIRating > 10:
                # Convert from 100-point scale to 1-10 scale
                if step.AIRating >= 90:
                    step.AIRating = 9
                elif step.AIRating >= 80:
                    step.AIRating = 8
                elif step.AIRating >= 70:
                    step.AIRating = 7
                elif step.AIRating >= 60:
                    step.AIRating = 6
                elif step.AIRating >= 50:
                    step.AIRating = 5
                elif step.AIRating >= 40:
                    step.AIRating = 4
                elif step.AIRating >= 30:
                    step.AIRating = 3
                elif step.AIRating >= 20:
                    step.AIRating = 2
                elif step.AIRating >= 10:
                    step.AIRating = 1
                else:
                    step.AIRating = 0
                print(f"⚠️ AI Rating was > 10, converted to: {step.AIRating}")
            
            # Fix FinalResult if it's coming as 0/1 instead of boolean
            if isinstance(step.FinalResult, int):
                original_value = step.FinalResult
                step.FinalResult = bool(step.FinalResult)
                print(f"⚠️ FinalResult was integer {original_value}, converted to boolean: {step.FinalResult}")
    
    # Convert the Pydantic model to JSON
    json_output = math_solution.model_dump_json(indent=2)
    return json_output,total_tokens



# if __name__ == "__main__":
#     combined_input = """
#     Resume data:
# {
#   "status": "success",
#   "resume_id": "bc4a647d-8056-478b-873d-d421d373126b",
#   "resume_data": {
#     "SuggestedRole": "Chief Architect or VP of Engineering",
#     "CandidateFullName": "Gagan Bajpai",
#     "EmailAddress": "",
#     "PhoneNumber": "",
#     "Skills": [
#       "Kubernetes",
#       "Model Design",
#       "Large Systems Design",
#       "Java",
#       "Spring"
#     ],
#     "Experience": [
#       {
#         "CompanyName": "Lenskart.com",
#         "Position": "Tech@lenskart-GM",
#         "Duration": {
#           "StartDate": "January 2022",
#           "EndDate": "April 2025"
#         },
#         "CompanyType": "Product",
#         "BusinessType": "B2C",
#         "NumberOfEmployees": "10,880 employees",
#         "Funding": "$220M",
#         "Location": "Hyderabad, Telangana, India"
#       },
#       {
#         "CompanyName": "Amazon",
#         "Position": "Sr./Software development Manager",
#         "Duration": {
#           "StartDate": "July 2015",
#           "EndDate": "December 2021"
#         },
#         "CompanyType": "Product",
#         "BusinessType": "B2C",
#         "NumberOfEmployees": "1.0M employees",
#         "Funding": "Public company",
#         "Location": "Hyderabad Area, India"
#       },
#       {
#         "CompanyName": "Times Internet Limited",
#         "Position": "Vice President /General Manager/Chief Architect",
#         "Duration": {
#           "StartDate": "February 2011",
#           "EndDate": "July 2015"
#         },
#         "CompanyType": "Product",
#         "BusinessType": "B2C",
#         "NumberOfEmployees": "10,000 employees",
#         "Funding": null,
#         "Location": "Noida"
#       },
#       {
#         "CompanyName": "Lime Labs India Pvt. Ltd.",
#         "Position": "Solution Architect",
#         "Duration": {
#           "StartDate": "September 2009",
#           "EndDate": "November 2010"
#         },
#         "CompanyType": "Product",
#         "BusinessType": "B2B",
#         "NumberOfEmployees": "250 employees",
#         "Funding": null,
#         "Location": ""
#       },
#       {
#         "CompanyName": "Times Business Solutions Limited",
#         "Position": "Solution Architect",
#         "Duration": {
#           "StartDate": "September 2008",
#           "EndDate": "August 2009"
#         },
#         "CompanyType": "Product",
#         "BusinessType": "B2C",
#         "NumberOfEmployees": "4,999 employees",
#         "Funding": "Public company",
#         "Location": ""
#       },
#       {
#         "CompanyName": "GlobalLogic",
#         "Position": "Module Lead/Sr. Lead Architect/Solution Architect/Consultant Architect",
#         "Duration": {
#           "StartDate": "January 2003",
#           "EndDate": "August 2008"
#         },
#         "CompanyType": "Service",
#         "BusinessType": "B2B",
#         "NumberOfEmployees": "15,000 employees",
#         "Funding": null,
#         "Location": ""
#       },
#       {
#         "CompanyName": "Churchill Insurance",
#         "Position": "Software Engineer/Analyst and Sr. Analyst",
#         "Duration": {
#           "StartDate": "May 2001",
#           "EndDate": "January 2003"
#         },
#         "CompanyType": "Banking",
#         "BusinessType": "Banking",
#         "NumberOfEmployees": "10,200 employees",
#         "Funding": "Public company",
#         "Location": ""
#       },
#       {
#         "CompanyName": "BondGlobe Inc.",
#         "Position": "Senior Web Developer",
#         "Duration": {
#           "StartDate": "July 2000",
#           "EndDate": "May 2001"
#         },
#         "CompanyType": "Product",
#         "BusinessType": "B2B",
#         "NumberOfEmployees": "500 employees",
#         "Funding": "$6M",
#         "Location": ""
#       },
#       {
#         "CompanyName": "IIS Infotech",
#         "Position": "System Associate",
#         "Duration": {
#           "StartDate": "April 1999",
#           "EndDate": "July 2000"
#         },
#         "CompanyType": "Service",
#         "BusinessType": "B2B",
#         "NumberOfEmployees": "4,999 employees",
#         "Funding": "Public company",
#         "Location": ""
#       },
#       {
#         "CompanyName": "Vam Organic Chemicals Ltd.",
#         "Position": "Sr. Engineer",
#         "Duration": {
#           "StartDate": "May 1995",
#           "EndDate": "August 1998"
#         },
#         "CompanyType": "Product",
#         "BusinessType": "B2B",
#         "NumberOfEmployees": null,
#         "Funding": "Public company",
#         "Location": ""
#       }
#     ],
#     "Education": [
#       {
#         "CollegeUniversity": "HBTI",
#         "CourseDegree": "BTech, Chemical Technology",
#         "GraduationYear": "1995"
#       },
#       {
#         "CollegeUniversity": "Indian Institute of Technology, Delhi",
#         "CourseDegree": "Post Graduate Diploma, Computer Science",
#         "GraduationYear": "1999"
#       }
#     ],
#     "StabilityAssessment": "Lenskart.com: 3.33 years, Amazon: 6.5 years, Times Internet Limited: 4.5 years, Lime Labs India Pvt. Ltd.: 1.25 years, Times Business Solutions Limited: 1 year, GlobalLogic: 5.67 years, Churchill Insurance: 1.67 years, BondGlobe Inc.: 0.92 years, IIS Infotech: 1.25 years, Vam Organic Chemicals Ltd.: 3.33 years"
#   },
#   "TotalYearsOfExperience": 29.58,
#   "upload_date": "2025-06-23 11:51:55",
#   "usage": {
#     "tokens": 5053,
#     "cost": 0.04041
#   }
# }

# Job Description data:

# {
#   "status": "success",
#   "jd_id": "b6cb5c9c-d09e-4cc7-98bf-3bc17d478820",
#   "jd_data": {
#     "CompanyName": "Quince",
#     "JobTitle": "Frontend Developer",
#     "JobLocation": "Not specified, assume remote or HQ location based on company context",
#     "RequiredSkills": {
#       "technical": [
#         "JavaScript",
#         "HTML",
#         "CSS",
#         "React Native",
#         "RESTful APIs",
#         "asynchronous programming",
#         "web security principles",
#         "performance optimization",
#         "debugging tools and techniques"
#       ]
#     },
#     "YearsOfExperienceRequired": "5+ years",
#     "EducationRequirements": "Bachelor's or Master's degree in Computer Science or a related field",
#     "CompanyTypePreference": "Product",
#     "BusinessTypePreference": "B2C",
#     "OtherImportantRequirements": [
#       "Experience in frontend development with a focus on React Native with Android or iOS",
#       "Strong problem-solving and analytical skills",
#       "Excellent communication and interpersonal skills",
#       "Ability to work independently and as part of a team"
#     ]
#   },
#   "upload_date": "2025-06-21 21:18:20",
#   "usage": {
#     "tokens": 1632,
#     "cost": 0.01304
#   }
# }






    
#     """
#     json_output,total_tokens = analyze_resume_and_jd(combined_input)
#     print(json_output)
#     print(total_tokens)