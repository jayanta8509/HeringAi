import os
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import json
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

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


def analyze_resume_and_jd(combined_input):

    prompt_template = """
    You are an expert recruitment assistant. Analyze how well the candidate matches the job description.
        Provide the following:

        1. AI Rating (1-10) - Calculate using detailed scoring breakdown (Total = 100 points, then convert to 1-10 scale):
           
           IMPORTANT: The final AIRating field MUST be between 0-10, NOT the raw score!
           
           First calculate the total points (0-100), then convert to 1-10 scale using the conversion table below.
           Scoring Categories:
           a) Company Type Match (30 points max):
              - 30 points: Perfect match (Product candidate experience + Product company posting JD, OR Service candidate experience + Service company posting JD)
              - 20 points: Mixed candidate experience (both Product + Service) matching either type of hiring company
              - 10 points: Partial mismatch (Product candidate experience + Service company posting JD, OR Service candidate experience + Product company posting JD)
              - 0 points: Complete mismatch or unclear company types
           
           b) Domain Match - Business Type (17 points max):
              - 17 points: Exact match (B2B to B2B, B2C to B2C)
              - 12 points: Compatible mixed models (B2C/B2B experience for B2B role)
              - 8 points: Somewhat relevant
              - 0 points: No relevance
           
           c) Keywords + Skill Set Match (22 points max):
              - 22 points: 80%+ skills match JD tech stack
              - 18 points: 60-79% skills match
              - 14 points: 40-59% skills match
              - 10 points: 20-39% skills match
              - 5 points: <20% skills match
              - 0 points: No relevant skills
           
           d) Role Match (10 points max):
              - 10 points: Exact role match (Frontend to Frontend, Backend to Backend)
              - 7 points: Related role (Full-stack to Frontend/Backend)
              - 3 points: Transferable role
              - 0 points: Fundamental mismatch
           
           e) Responsibilities Match (15 points max):
              - 15 points: Candidate's projects/work clearly maps to JD expectations
              - 10 points: Partially relevant experience
              - 5 points: Loosely related
              - 0 points: No relevant project experience
           
           f) College Prestige (3 points max):
              - 3 points: Top-tier (IIT, NIT, MIT, BITS Stanford, etc.)
              - 2 points: Good tier (State universities, reputable colleges)
              - 1 point: Average tier
              - 0 points: Unknown/lesser-known institutions
           
           g) Awards & Recognition (3 points max):
              - 3 points: Relevant certifications, awards, recognitions related to JD
              - 1-2 points: Some certifications/awards
              - 0 points: None mentioned
           
           MANDATORY CONVERSION TABLE - USE THIS TO CONVERT TO FINAL AIRating:
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
           
           EXAMPLE: If you calculate 21 total points, the AIRating field should be 2, NOT 21!
        2. Whether the candidate should be shortlisted (Yes/No)
           - Should be "No" if there's a fundamental role mismatch (backend candidate for frontend role or vice versa)
           - Rating should be ≤4 if there's a fundamental role mismatch (backend candidate for frontend job)
           - Rating should consider skill relevance percentage: if <50% skills match, rating should be ≤5

        6. Analysis of each company in the candidate's resume:
           - Company name
           - Company type (Product/Service)
           - Industry sector
           - Business model (B2B/B2C)
           - Any notable achievements
        7. Education assessment:
           - College/University assessment
           - Course relevance
        8. Anything missing as per expectations in the JD
           - Include fundamental role mismatches 
         - Include responsibilities mismatches
           - Highlight missing core skills for the specific role
           - Note experience level gaps
        9. Overall recommendation (detailed summary in 3-4 lines)
           - CRITICAL: Write in perfect, grammatically correct English with proper sentence structure.
           - Use professional, clear, and concise language without any grammatical errors.
           - MUST include work experience breakdown by company type as percentages.
           - Calculate total years of experience and show percentage split: "X% service-based company, Y% product-based company, and Z% banking company based on work experience."
           - For product companies, further divide experience into business models:
            - Calculate the percentage of B2B vs B2C experience **based on the duration spent at each product company.**
            - If the product experience includes only B2B or B2C, mention that.
            - If it includes both, state the exact percentage like: “Product company work includes 63% B2B and 37% B2C.”
          - For service based companies business type is always considered as Services company.
          - For banking companies, business type is always considered Banking.
          - Ensure smooth flow, accurate calculations, proper punctuation, and subject-verb agreement.
          - Final Evaluation (based on FinalResult boolean):
            - If FinalResult = true, add: "This candidate is suitable for this role."
            - If FinalResult = false, add: "This candidate is not suitable for this role."
          - Example: "The candidate has 6 years of total experience: 33% in service-based companies, 50% in product-based companies, and 17% in banking companies. The product company experience includes 34% B2B and 66% B2C, offering well-rounded exposure across both segments. The service company experience contributes solid B2B operational knowledge, and the banking experience adds domain-specific expertise. This candidate is suitable for this role."

        10. Candidate status prediction:
           - Should be AI shortlisted (Yes/No)
           - Should be internally shortlisted (Yes/No)
           - Ready for interview process (Yes/No)
           - Final result prediction (Selected/Rejected)
              - Based on the candidate's experience, skills, education, suggested role, total years of experience, and alignment with the job description (JD) requirements, determine whether the candidate is a good fit for the role.
              - "Selected" If the candidate is a good fit, return true.
              - "Rejected" If the candidate is not a good fit, return false.
           - Likelihood of joining if offered (High/Medium/Low)
        
        Format your response as a JSON object with the following structure:
        {
          "AIRating": number,
          "ShouldBeShortlisted": "Yes/No",
          "CompanyAnalysis": [
            {
              "CompanyName": "string",
              "CompanyType": "string",
              "IndustrySector": "string",
              "BusinessModel": "string (B2B/B2C)",
              "NotableAchievements": "string"
            }
          ],
          "EducationAssessment": {
            "UniversityAssessment": "string",
            "CourseRelevance": "string"
          },
          "MissingExpectations": ["string"],
          "OverallRecommendation": "string (detailed summary in 2-3 lines)",
          "AIShortlisted": "Yes/No",
          "InternalShortlisted": "Yes/No",
          "InterviewInProcess": "Yes/No",
          "FinalResult": "boolean (true for Selected, false for Rejected)",
          "CandidateJoined": "Yes/No/Unknown"
        }
        
        For the candidate status prediction:
        - AIRating MUST be a number from 0-10 (converted from 100-point scale, NOT the raw score!)
        - AIShortlisted should be "Yes" if the AIRating is 7 or higher, otherwise "No"
        - InternalShortlisted should be your recommendation based on the candidate's fit
        - InterviewInProcess should be "Yes" if you recommend they proceed to interviews
        - FinalResult should be true if they're an excellent match, false if poor match.
        - CandidateJoined should be your prediction of whether they'd join if offered
        
        CRITICAL: Double-check that your AIRating is 0-10, not the raw points!

        """

    completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
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