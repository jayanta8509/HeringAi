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
    SuggestedRole: str
    AIRating: int
    ShouldBeShortlisted: str
    CompanyTypeMatch: str
    BusinessTypeMatch: str
    StabilityAssessment: list[str]
    CompanyAnalysis: list[CompanyAnalysisItem]
    EducationAssessment: EducationAssessment
    MissingExpectations: list[str]
    OverallRecommendation: str
    AIShortlisted: str
    InternalShortlisted: str
    InterviewInProcess: str
    FinalResult: str
    CandidateJoined: str

class resume_data(BaseModel):
    steps: list[Step]


def analyze_resume_and_jd(combined_input):

    prompt_template = """
    You are an expert recruitment assistant. Analyze how well the candidate matches the job description.
        Provide the following:
        1. Suggested role for the candidate (e.g., Frontend, Backend, DevOps, etc.)
           - Analyze the candidate's PRIMARY skill set and experience to determine their core expertise
           - Backend indicators: Java, Spring, Node.js, Python, databases (MySQL, PostgreSQL), AWS services, microservices, APIs
           - Frontend indicators: React, Angular, Vue, HTML, CSS, JavaScript (as primary skills), UI/UX tools
           - Testing indicators : Manual testing and automation testing selenium,qtp,Appium,SDET
           - Devops indicators : AWS Engineer, Azure Engineer, cloud Engineer, Devops Engineer 
           - If candidate is primarily backend but job requires frontend: Suggest "Backend" and note the mismatch

        2. AI Rating (1-10) - Calculate using detailed scoring breakdown (Total = 100 points, then convert to 1-10 scale):
           
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
        3. Whether the candidate should be shortlisted (Yes/No)
           - Should be "No" if there's a fundamental role mismatch (backend candidate for frontend role or vice versa)
           - Rating should be ≤4 if there's a fundamental role mismatch (backend candidate for frontend job)
           - Rating should consider skill relevance percentage: if <50% skills match, rating should be ≤5
        4. Company type match (Product/Service)
        5. Business type match (B2B/B2C/combinations - consider partial matches for mixed models)
        6. Stability assessment (company-wise tenure duration as an array):
        - For each unique company in the candidate's experience, sum the total tenure duration across all stints at that company.
        - Provide the company name and the total tenure duration in years (rounded to two decimal places), e.g., "Amazon: 1.16 years".
        - Output the result as an array of strings, one per unique company, in the order they first appear in the candidate's experience.
        - Do not include any extra commentary or summary—just the array of company-wise total tenure durations.
        7. Analysis of each company in the candidate's resume:
           - Company name
           - Company type (Product/Service)
           - Industry sector
           - Business model (B2B/B2C)
           - Any notable achievements
        8. Education assessment:
           - College/University assessment
           - Course relevance
        9. Anything missing as per expectations in the JD
           - Include fundamental role mismatches 
         - Include responsibilities mismatches
           - Highlight missing core skills for the specific role
           - Note experience level gaps
        10. Overall recommendation (detailed summary in 3-4 lines)
        11. Candidate status prediction:
           - Should be AI shortlisted (Yes/No)
           - Should be internally shortlisted (Yes/No)
           - Ready for interview process (Yes/No)
           - Final result prediction (Selected/Rejected/Pending)
           - Likelihood of joining if offered (High/Medium/Low)
        
        IMPORTANT: For business type matching:
        - B2C/B2B experience is compatible with B2B requirements
        - B2B/B2C experience is compatible with B2C requirements 
         - Services experience is compatible with Services requirements        
        COMPANY TYPE CLASSIFICATION GUIDANCE:
        For accurate company type classification, use the following guidelines:
        - Amazon, Google, Microsoft, Apple, Meta, Netflix: Product companies
        - Moneyview: Product company (fintech with lending products and financial services platform)
        - Flipkart, Zomato, Paytm, Swiggy: Product companies (platform/app-based)
        - TCS, Tata Consultancy Services, Infosys, Wipro, Accenture, Cognizant: Service companies
        - Banks (HDFC, ICICI, SBI), unless they have significant product divisions: Banking companies
        - Startups with apps/platforms/SaaS products: Product companies        
        When determining CompanyTypeMatch:
        - If all companies in candidate's experience are Product companies: "Product"
        - If all companies in candidate's experience are Service companies: "Service" 
        - If candidate has mixed experience (both Product and Service): "Product/Service"
        
        CRITICAL: Analyze the CompanyType field in the CompanyAnalysis section you generate. 
        If ALL companies show CompanyType as "Product", then CompanyTypeMatch MUST be "Product".
        If ALL companies show CompanyType as "Service", then CompanyTypeMatch MUST be "Service".
        Only use "Product/Service" when there's a genuine mix of Product and Service companies.
        
        Format your response as a JSON object with the following structure:
        {
          "SuggestedRole": "string",
          "AIRating": number,
          "ShouldBeShortlisted": "Yes/No",
          "CompanyTypeMatch": "string (MUST be 'Product' if all CompanyAnalysis entries are Product type, 'Service' if all are Service type, 'Product/Service' only for mixed experience)",
          "BusinessTypeMatch": "string (explain compatibility for mixed models)",
          "StabilityAssessment": ["string (company name and total tenure duration, e.g., 'Amazon: 1.16 years')", ...],
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
          "FinalResult": "Selected/Rejected/Pending",
          "CandidateJoined": "Yes/No/Unknown"
        }
        
        For the candidate status prediction:
        - AIRating MUST be a number from 0-10 (converted from 100-point scale, NOT the raw score!)
        - AIShortlisted should be "Yes" if the AIRating is 7 or higher, otherwise "No"
        - InternalShortlisted should be your recommendation based on the candidate's fit
        - InterviewInProcess should be "Yes" if you recommend they proceed to interviews
        - FinalResult should be "Selected" if they're an excellent match, "Rejected" if poor match, "Pending" if moderate
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
    
    # Convert the Pydantic model to JSON
    json_output = math_solution.model_dump_json(indent=2)
    return json_output,total_tokens

