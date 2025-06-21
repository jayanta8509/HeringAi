import os
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

class RequiredSkills(BaseModel):
    technical: list[str]

class Step(BaseModel):
    CompanyName: str | None = None
    JobTitle: str
    JobLocation: str
    RequiredSkills: RequiredSkills
    YearsOfExperienceRequired: str
    EducationRequirements: str
    CompanyTypePreference: str | None = None
    BusinessTypePreference: str | None = None
    OtherImportantRequirements: list[str]

class jd_data(BaseModel):
    steps: list[Step]

def analyze_jd(input_question):

    prompt_template = """ Your role is expert-JD-analyst and your task is to extract key fields from a Job Description and return a JSON object.
        DATA TO EXTRACT
        1. CompanyName (from email domains, headers, “About us”, etc.) or null
        2. JobTitle
        3. JobLocation
        4. RequiredSkills → { technical }
        5. YearsOfExperienceRequired
        6. EducationRequirements
        7. CompanyTypePreference → Product | Service | Banking | null (infer)
        8. BusinessTypePreference → B2B | B2C | Services | null (infer)
        9. PreferredStability → string or null (e.g., ≥ 2 yrs per company;)
        10. OtherImportantRequirements → list
        
        Format your response as a JSON object with the following structure:
        {
          "CompanyName": "string (extract from any source in JD) or null if genuinely not found",
          "JobTitle": "string",
          "JobLocation": "string",
          "RequiredSkills": {
            "technical": ["skill1", "skill2", ...]
          },
          "YearsOfExperienceRequired": "string",
          "EducationRequirements": "string",
          "CompanyTypePreference": "Product/Service/Banking (infer if not explicit) or null",
          "BusinessTypePreference": "B2B/B2C/Services (infer if not explicit) or null",
          "OtherImportantRequirements": ["requirement1", "requirement2", ...]
        }
        """

    completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": input_question}
    ],
    response_format=jd_data,
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
        math_solution = jd_data(steps=math_reasoning.parsed.steps)
    
    # Convert the Pydantic model to JSON
    json_output = math_solution.model_dump_json(indent=2)
    return json_output,total_tokens


# if __name__ == "__main__":
#     input_question = """
#     OUR STORY Quince was started to challenge the existing idea that nice things should cost a  lot. Our mission was simple: create an item of equal or greater quality than the  leading luxury brands and sell them at a much lower price. OUR VALUES Customer First. Customer satisfaction is our highest priority. High Quality. True quality is a combination of premium materials and high  production standards that everyone can feel good about. Essential design. We don't chase trends, and we don't sell everything. We're  expert curators that find the very best and bring it to you at the lowest  prices. Always a better deal. Through innovation and real price transparency we want to  offer the best deal to both our customers and our factory partners. Environmentally and Socially conscious. We're committed to sustainable materials  and sustainable production methods. That means a cleaner environment and fair  wages for factory workers. OUR TEAM AND SUCCESS Quince is a retail and technology company co
# - founded by a team that has  extensive experience in retail, technology and building early
# - stage companies.  You'll work with a team of world
# - class talent from Stanford GSB, Google, D.E.  Shaw, Stitch Fix, Urban Outfitters, Wayfair, McKinsey, Nike etc. Key Responsibilities ·         Design, develop, and maintain frontend applications using modern  mobile and web technologies. ·         Collaborate with cross
# - functional teams to define requirements and  implement solutions. ·         Write clean, efficient, and well
# - documented code. ·         Debug and resolve technical issues in a timely manner. ·         Participate in code reviews and provide constructive feedback to  peers. ·         Develop and maintain test cases to ensure high
# - quality code. ·         Stay up
# - to
# - date with emerging trends in frontend development and  recommend technologies and approaches that can improve our applications. ·         Mentor and guide junior engineers on the team. Qualification ·         Bachelor's or Master's degree in Computer Science or a related field. ·         At least 5+ years of professional experience in frontend development. ·         Strong proficiency in JavaScript, HTML, CSS, and related web  technologies ·         Experience with modern frontend framework React Native with Android or  IOS ·         Proficient in using debugging tools and techniques ·         Strong understanding of RESTful APIs and asynchronous programming.  ·         Knowledge of web security principles and techniques. ·         Experience with performance optimization techniques. ·         Strong problem
# - solving and analytical skills. ·         Excellent communication and interpersonal skills. ·         Ability to work independently as well as part of a team.
#     """
#     json_output,total_tokens = analyze_jd(input_question)
#     print(json_output)
#     print(total_tokens)




