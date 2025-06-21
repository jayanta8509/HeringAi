import os
import re
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

class Duration(BaseModel):
    StartDate: str
    EndDate: str

class ExperienceItem(BaseModel):
    CompanyName: str
    Position: str
    Duration: Duration
    CompanyType: str
    BusinessType: str
    NumberOfEmployees: str | None = None
    Funding: str | None = None
    Location: str

class EducationItem(BaseModel):
    CollegeUniversity: str
    CourseDegree: str
    GraduationYear: str

class Step(BaseModel):
    CandidateFullName: str
    EmailAddress: str
    PhoneNumber: str
    Skills: list[str]
    Experience: list[ExperienceItem]
    Education: list[EducationItem]
    StabilityAssessment: str

class resume_data(BaseModel):
    steps: list[Step]


def search_company_info_with_web_search(company_name: str) -> dict:
    """
    Search for company information using OpenAI's Web Search
    
    Args:
        company_name: Name of the company to search
        
    Returns:
        Dictionary with NumberOfEmployees and Funding information
    """
    company_info = {
        "NumberOfEmployees": None,
        "Funding": None
    }
    
    try:
        # Search for employee count
        employee_prompt = f"""
        Search for information about {company_name} company and find:
        1. How many employees does {company_name} have?
        2. What is the company's workforce size or headcount?
        
        Please provide the specific number of employees if available, and format it as "X employees" where X is the number.
        If it's a large company, you can use formats like "1.5M employees" for 1.5 million employees.
        """
        
        employee_response = client.responses.create(
            model="gpt-4.1",
            tools=[{
                "type": "web_search_preview",
                "search_context_size": "medium"
            }],
            input=employee_prompt
        )
        
        # Extract text from the response structure
        employee_text = ""
        for output in employee_response.output:
            if output.type == "message" and output.content:
                for content_item in output.content:
                    if content_item.type == "output_text":
                        employee_text = content_item.text
                        break
        employee_count = extract_employee_count(employee_text)
        if employee_count:
            company_info["NumberOfEmployees"] = employee_count
        
        # Search for funding information
        funding_prompt = f"""
        Search for information about {company_name} company and find:
        1. How much funding has {company_name} raised?
        2. What is {company_name}'s valuation?
        3. Is {company_name} a public company or private company?
        4. What type of funding rounds has {company_name} completed?
        
        Please provide specific funding amounts in millions or billions (e.g., $75M, $2.1B) or indicate if it's a public company.
        """
        
        funding_response = client.responses.create(
            model="gpt-4.1", 
            tools=[{
                "type": "web_search_preview",
                "search_context_size": "medium"
            }],
            input=funding_prompt
        )
        
        # Extract text from the response structure
        funding_text = ""
        for output in funding_response.output:
            if output.type == "message" and output.content:
                for content_item in output.content:
                    if content_item.type == "output_text":
                        funding_text = content_item.text
                        break
        funding_info = extract_funding_info(funding_text)
        if funding_info:
            company_info["Funding"] = funding_info
        
        return company_info
        
    except Exception as e:
        print(f"Error searching for {company_name}: {str(e)}")
        return company_info


def extract_employee_count(content: str) -> str | None:
    """Extract employee count from search content"""
    patterns = [
        # Standard patterns
        r'(\d+(?:,\d+)*)\s*employees',
        r'(\d+(?:,\d+)*)\s*people',
        r'workforce\s*of\s*(\d+(?:,\d+)*)',
        r'employs\s*(?:about|over|approximately)?\s*(\d+(?:,\d+)*)',
        r'has\s*(?:about|over|approximately)?\s*(\d+(?:,\d+)*)\s*(?:employees|workers|people)',
        r'headcount[:\s]*(\d+(?:,\d+)*)',
        r'total\s*employees[:\s]*(\d+(?:,\d+)*)',
        
        # Million patterns for large companies
        r'(\d+(?:\.\d+)?)\s*million\s*employees',
        r'over\s*(\d+(?:\.\d+)?)\s*million\s*(?:employees|people)',
        
        # Alternative patterns
        r'(\d+(?:,\d+)*)\s*full-time\s*employees',
        r'global\s*workforce\s*of\s*(\d+(?:,\d+)*)',
        r'total\s*workforce[:\s]*(\d+(?:,\d+)*)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            count = match.group(1)
            try:
                # Handle million patterns
                if 'million' in pattern:
                    num = float(count)
                    if 0.1 <= num <= 10:  # 0.1M to 10M employees
                        return f"{num}M employees"
                else:
                    num = int(count.replace(',', ''))
                    if 10 <= num <= 10000000:  # Reasonable range
                        return f"{count} employees"
            except:
                continue
    return None


def extract_funding_info(content: str) -> str | None:
    """Extract funding information from search content"""
    # Check for public company first
    public_patterns = [
        r'public\s*company',
        r'publicly\s*traded',
        r'listed\s*on\s*(?:nasdaq|nyse|stock\s*exchange)',
        r'stock\s*ticker',
        r'shares\s*trade',
        r'market\s*capitalization'
    ]
    
    for pattern in public_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return "Public company"
    
    # Extract funding amounts
    funding_patterns = [
        r'\$(\d+(?:\.\d+)?)\s*(million|billion|m|b)\s*(?:in\s*)?(?:funding|investment|valuation|raised)',
        r'raised\s*\$(\d+(?:\.\d+)?)\s*(million|billion|m|b)',
        r'funding\s*of\s*\$(\d+(?:\.\d+)?)\s*(million|billion|m|b)',
        r'valuation\s*of\s*\$(\d+(?:\.\d+)?)\s*(million|billion|m|b)',
        r'series\s*[a-z]\s*\$(\d+(?:\.\d+)?)\s*(million|billion|m|b)',
        r'worth\s*\$(\d+(?:\.\d+)?)\s*(million|billion|m|b)',
        r'valued\s*at\s*\$(\d+(?:\.\d+)?)\s*(million|billion|m|b)',
        r'\$(\d+(?:\.\d+)?)\s*(million|billion|m|b)\s*round',
        r'investment\s*of\s*\$(\d+(?:\.\d+)?)\s*(million|billion|m|b)',
        r'total\s*funding[:\s]*\$(\d+(?:\.\d+)?)\s*(million|billion|m|b)',
        r'has\s*raised\s*\$(\d+(?:\.\d+)?)\s*(million|billion|m|b)'
    ]
    
    for pattern in funding_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            amount = match.group(1)
            unit = match.group(2).lower()
            if unit in ['m', 'million']:
                return f"${amount}M"
            elif unit in ['b', 'billion']:
                return f"${amount}B"
    
    return None


def enrich_company_data(experience_list: list[ExperienceItem]) -> list[ExperienceItem]:
    """
    Enrich company data by searching for missing NumberOfEmployees and Funding information
    
    Args:
        experience_list: List of experience items from resume parsing
        
    Returns:
        Enriched list of experience items
    """
    enriched_experience = []
    
    for exp in experience_list:
        # Create a copy of the experience item
        enriched_exp = exp.model_copy()
        
        # Check if we need to search for missing information
        needs_search = (exp.NumberOfEmployees is None or exp.Funding is None)
        
        if needs_search:
            print(f"Searching for company info: {exp.CompanyName}")
            company_info = search_company_info_with_web_search(exp.CompanyName)
            
            # Update missing fields
            if exp.NumberOfEmployees is None and company_info.get("NumberOfEmployees"):
                enriched_exp.NumberOfEmployees = company_info["NumberOfEmployees"]
                print(f"✅ Found employee count for {exp.CompanyName}: {company_info['NumberOfEmployees']}")
            
            if exp.Funding is None and company_info.get("Funding"):
                enriched_exp.Funding = company_info["Funding"]
                print(f"✅ Found funding info for {exp.CompanyName}: {company_info['Funding']}")
        
        enriched_experience.append(enriched_exp)
    
    return enriched_experience


def analyze_resume(input_question):

    prompt_template = """ Your role is expert-resume-parser and your task is to extract key fields from a resume and return a JSON object.
                        1. CandidateFullName
                        2. EmailAddress
                        3. PhoneNumber
                        4. Skills → list up to 5 core technical skills exactly as written
                        5. Experience → array; for each job include
                        • CompanyName (use email domains, LinkedIn URLs, or text clues)
                        • Position
                        • Duration → { StartDate, EndDate } (copy dates exactly)
                        • CompanyType → Product| Service | Banking (infer if missing)
                        • BusinessType → B2B | B2C | Services (infer if missing)
                        • NumberOfEmployees → string or null (only if résumé states it)
                        • CompanyRevenue → string or null (only if résumé states it)
                        • Funding → string or null (only if résumé states it)
                        • Location → city / country or null
                        6. Education → array of { CollegeUniversity, CourseDegree, GraduationYear }
                        7. StabilityAssessment → one sentence that sums average tenure
                        INFERENCE GUIDELINES
                        • Amazon, Google, Microsoft, Flipkart, Paytm = Product
                        • TCS, Infosys, Wipro, Accenture = Service
                        • Banks (HDFC, SBI, JPMorgan) = Banking
                        • E-commerce apps → Product &amp; B2C
                        • SaaS platforms → Product &amp; B2B
                        Use null when data is truly absent.

        
        FORMAT GUIDELINES: 
        Format your response as a JSON object with the following structure:
        {
          "CandidateFullName": "string",
          "EmailAddress": "string",
          "PhoneNumber": "string",
          "Skills": ["skill1", "skill2", ...],
          "Experience": [
            {
              "CompanyName": "string (extract carefully from any source)",
              "Position": "string",
              "Duration": {
                "StartDate": "string (EXACT as in resume)",
                "EndDate": "string (EXACT as in resume)"
              },
              "CompanyType": "Product/Service (infer if not explicit)",
              "BusinessType": "B2B/B2C/B2C/B2B/B2B2C (infer if not explicit)",
              "NumberOfEmployees": "string or null",
              "Funding": "string or null",
              "Location": "string"
            }
          ],
          "Education": [
            {
              "CollegeUniversity": "string",
              "CourseDegree": "string",
              "GraduationYear": "string"
            }
          ],
          "StabilityAssessment": "string"
        }

        """

    completion = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",
    messages=[
        {"role": "system", "content": prompt_template},
        {"role": "user", "content": input_question}
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
        parsed_data = resume_data(steps=math_reasoning.parsed.steps)
        
        # Check if any companies have null NumberOfEmployees or Funding fields
        needs_enrichment = False
        for step in parsed_data.steps:
            for exp in step.Experience:
                if exp.NumberOfEmployees is None or exp.Funding is None:
                    needs_enrichment = True
                    break
            if needs_enrichment:
                break
        
        # Automatically enrich company data with web search if null fields found
        if needs_enrichment:
            print("🔍 Found null company data. Automatically enriching with web search...")
            for step in parsed_data.steps:
                step.Experience = enrich_company_data(step.Experience)
            print("✅ Company data enrichment completed")
        else:
            print("ℹ️ All company data already populated. Skipping web search.")
        
        math_solution = parsed_data
    
    # Convert the Pydantic model to JSON
    json_output = math_solution.model_dump_json(indent=2)
    return json_output,total_tokens

