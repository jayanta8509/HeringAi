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
    FinalResult: bool
    CandidateJoined: str

class resume_data(BaseModel):
    steps: list[Step]


def analyze_resume_and_jd(combined_input):

    prompt_template = """Your role is expert-recruitment-assistant and your task is to compare the parsed résumé JSON to the parsed JD JSON and produce a single
        STEP 1 Pick SuggestedRole
        • Frontend → React, Angular, Vue, UI/UX
        • Backend → Java, Spring, Node.js, .NET, Python, Databases
        • FullStack → strong mix of both
        • DevOps, Testing, etc. as obvious
        STEP 2 Score the match (0-100 pts)
        a) CompanyTypeMatch 30 pts
        b) BusinessTypeMatch 17 pts
        c) SkillMatch 22 pts (exact %-match to JD technical list)
        d) RoleMatch 10 pts
        e) ResponsibilitiesMatch 15 pts (project vs JD duties)
        f) CollegePrestige 3 pts

        g) AwardsCerts 3 pts
        Convert to AIRating 0-10 (divide by 10, round 1 decimal).
        STEP 3 Decide ShouldBeShortlisted (Yes/No).
        Fundamental role or &lt;50 % skill match ⇒ “No”.
        STEP 4 Generate StabilityAssessment → array: &quot;Amazon: 1.60 yrs&quot;, keep
        candidate order.
        STEP 5 CompanyAnalysis → for each unique company:
        { CompanyName, CompanyType, IndustrySector, BusinessModel,
        NotableAchievements }
        STEP 6 EducationAssessment → 2 short fields.
        STEP 7 List MissingExpectations.
        STEP 8 Give OverallRecommendation (2-3 lines, plain English).
        STEP 9 Predict candidate status
        • AIShortlisted Yes if AIRating ≥ 7
        • InternalShortlisted Yes/No (your call)
        • InterviewInProcess Yes/No
        • FinalResult Selected | Rejected
        • CandidateJoined High | Medium | Low

        
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
          "FinalResult": "boolean (true for Selected, false for Rejected)",
          "CandidateJoined": "Yes/No/Unknown"
        }
        
        For the candidate status prediction:
        - AIRating MUST be a number from 0-10 (converted from 100-point scale, NOT the raw score!)
        - FinalResult should be true if they're an excellent match, false if poor match.
        
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

