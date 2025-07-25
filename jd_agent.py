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

    prompt_template = """ 
    You are an expert job description analyst. Extract the following information:
        1. Company name (LOOK CAREFULLY - check email domains, headers, footers, "About us" sections, contact info, company references, brand mentions)
        2. Job title
        3. Job location (city, country, remote status)
        4. Required skills (technical and soft skills)
        5. Years of experience required
        6. Education requirements
        7. Company type preference (Product (B2B or B2C) /Service - if not explicitly mentioned, infer from company name and context)
        8. Business type preference (B2B/B2C/combinations like B2C/B2B - if not mentioned, infer from company name and job context)
        9. Other important requirements
        
        IMPORTANT INSTRUCTIONS FOR COMPANY DETECTION:
        - Look for company names in email addresses (e.g., @tcs.com suggests TCS)
        - Check headers, footers, letterheads, or signatures
        - Look for phrases like "About [Company]", "Join [Company]", "At [Company]"
        - Check for brand names, subsidiary names, or parent company references
        - If you find a company name, try to infer the company type and business type based on your knowledge
        
        COMPANY TYPE INFERENCE RULES:
        - TCS, Tata Consultancy Services, Infosys, Wipro, Accenture, Cognizant = Service companies
        - Amazon, Google, Microsoft, Apple, Meta, Netflix = Product companies
        - Banks = Banking  companies
        - Software products, E-commerce, SaaS platforms = Product companies
        
        BUSINESS TYPE INFERENCE RULES:
        - IT Services companies (TCS, Infosys) = Services 
        - E-commerce (Amazon retail) = B2C
        - Enterprise software (Microsoft) = B2B
        - Social media (Meta) = B2C
        - Consumer products (Apple) = B2B
        
        For business type, use these formats:
        - B2B: Primarily business-to-business
        - B2C: Primarily business-to-consumer
        - Services : Services companies from technology point of view

        
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
          "CompanyTypePreference": "Product/Service (infer if not explicit) or null",
          "BusinessTypePreference": "B2B/B2C/Services (infer if not explicit) or null",
          "OtherImportantRequirements": ["requirement1", "requirement2", ...]
        }
        
        EXAMPLES OF INFERENCE:
        - If JD mentions "tcs.com" email → CompanyName: "TCS", CompanyTypePreference: "Service"
        - If JD mentions "amazon.com" → CompanyName: "Amazon", CompanyTypePreference: "Product", BusinessTypePreference: "B2C "
        - If JD mentions "google.com" → CompanyName: "Google", CompanyTypePreference: "Product", BusinessTypePreference: "B2C "

        """

    completion = client.beta.chat.completions.parse(
    model="gpt-4.1-nano-2025-04-14",
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




