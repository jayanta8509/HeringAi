import os
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import json
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
from langfuse import observe

class CompanyAnalysisItem(BaseModel):
    CompanyName: str
    CompanyType: str
    IndustrySector: str
    BusinessModel: str
    NotableAchievements: str

class EducationAssessment(BaseModel):
    UniversityAssessment: str
    CourseRelevance: str

class Step(BaseModel):
    AIRating: int
    ShouldBeShortlisted: str
    CompanyAnalysis: list[CompanyAnalysisItem]
    EducationAssessment: EducationAssessment
    MissingExpectations: list[str]
    OverallRecommendation: str
    AIShortlisted: str
    InternalShortlisted: str
    InterviewInProcess: str
    FinalResult: bool
    CandidateJoined: str

class resume_data(BaseModel):
    steps: list[Step]

@observe(name="analyze_resume_and_jd")
def analyze_resume_and_jd(combined_input):

    prompt_template = """Analyze candidate-job description alignment and provide comprehensive evaluation.

## Analysis Framework:

### 1. AI Rating Calculation (0-10 scale)
Calculate total points (0-100), then convert using conversion table below.

#### Scoring Components:
**a) Company Type Match (30 points max):**
- 30 points: Perfect alignment (Product↔Product OR Service↔Service)
- 20 points: Mixed experience matching hiring company type
- 10 points: Misaligned (Product→Service OR Service→Product)
- 0 points: Complete mismatch/unclear types

**b) Business Type Match (17 points max):**
- 17 points: Exact match (B2B↔B2B, B2C↔B2C)
- 12 points: Compatible models (B2C/B2B experience for B2B role)
- 8 points: Somewhat relevant
- 0 points: No relevance

**c) Technical Skills Match (22 points max):**
- 22 points: ≥80% skills match JD requirements
- 18 points: 60-79% skills match
- 14 points: 40-59% skills match
- 10 points: 20-39% skills match
- 5 points: <20% skills match
- 0 points: No relevant skills

**d) Role Alignment (10 points max):**
- 10 points: Exact match (Frontend↔Frontend, Backend↔Backend)
- 7 points: Related (Full-stack↔Frontend/Backend)
- 3 points: Transferable skills
- 0 points: Fundamental mismatch

**e) Responsibilities Match (15 points max):**
- 15 points: Clear project/work mapping to JD expectations
- 10 points: Partially relevant experience
- 5 points: Loosely related
- 0 points: No relevant project experience

**f) Education Quality (3 points max):**
- 3 points: Top-tier (IIT, NIT, MIT, BITS, Stanford)
- 2 points: Good tier (reputable universities)
- 1 point: Average tier
- 0 points: Unknown/lesser institutions

**g) Certifications/Awards (3 points max):**
- 3 points: Relevant certifications/awards for role
- 1-2 points: Some certifications/awards
- 0 points: None mentioned

#### Conversion Table (MANDATORY):
- 90-100 points → AI Rating 9-10
- 80-89 points → AI Rating 8
- 70-79 points → AI Rating 7
- 60-69 points → AI Rating 6
- 50-59 points → AI Rating 5
- 40-49 points → AI Rating 4
- 30-39 points → AI Rating 3
- 20-29 points → AI Rating 2
- 10-19 points → AI Rating 1
- 0-9 points → AI Rating 0

**CRITICAL: Final AIRating must be 0-10, NOT raw score!**

### 2. Shortlisting Decision
- "No" for fundamental role mismatches
- Rating ≤4 for backend candidate applying to frontend role
- Rating ≤5 if <50% skills match

### 3. Company Analysis Requirements
For each company in candidate's experience:
- Company name and type (Product/Service/Banking)
- Industry sector and business model (B2B/B2C)
- Notable achievements or recognitions

### 4. Education Assessment
- University/college evaluation and reputation
- Course relevance to job requirements

### 5. Gap Analysis
Identify missing requirements:
- Fundamental role mismatches
- Responsibility gaps
- Missing core technical skills
- Experience level shortfalls

### 6. Overall Recommendation Format
**Requirements:**
- Professional, grammatically correct English
- Include experience breakdown by company type percentages
- Calculate: "X% service-based, Y% product-based, Z% banking"
- For product companies: specify B2B vs B2C percentage breakdown
- Final assessment: "This candidate is [suitable/not suitable] for this role"

**Example:** "The candidate has 6 years total experience: 33% service-based companies, 50% product-based companies, 17% banking companies. Product experience includes 34% B2B and 66% B2C exposure. This candidate is suitable for this role."

### 7. Status Predictions
- AI Shortlisted: "Yes" if rating ≥7, otherwise "No"
- Internal shortlisting recommendation
- Interview readiness assessment
- Final result: true (Selected) or false (Rejected)
- Joining likelihood: High/Medium/Low


**VERIFICATION:** Ensure AIRating is 0-10 scale, not raw 100-point score!"""

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