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
    SuggestedRole: str
    CandidateFullName: str
    EmailAddress: str
    PhoneNumber: str
    Skills: list[str]
    Experience: list[ExperienceItem]
    Education: list[EducationItem]
    StabilityAssessment: list[str]
    AverageStability: str
    CompanyTypeMatch: str
    BusinessTypeMatch: str
    ComplexWorkExperience: bool

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

    prompt_template = """ You are an expert resume parser. Extract the following information from the resume:
        1. Candidate's full name
        2. Email address (check personal email, work emails, LinkedIn profiles)
        3. Phone number
        4. Skills (list all technical skills that base on resume maximum 5 skills)
        5. For each company experience:
           - Company name (LOOK CAREFULLY - check email domains, LinkedIn URLs, official company names, subsidiaries)
           - Position/role
           - Duration (specify EXACT start date and end date in the same format they appear in the resume)
           - Whether it's a product or service or Banking company (infer from company name and context if not explicit)
           - Business type :
              - If Company Type is Service, then Business Type is always B2B
              - If Company Type is Product, then Business Type can be B2B, B2C, or both – infer based on the nature of the product, customers, or company description
              - If Company Type is Banking, then Business Type is always Banking
           - Number of employees (if mentioned or can be inferred from company knowledge)
           - Funding received and type of funding (if mentioned)
           - Company main location
        6. Education details:
           - College/University name
           - Course/degree
           - Graduation year
        7. Stability assessment (company-wise tenure duration as an array):
           - For each unique company in the candidate's experience, sum the total tenure duration across all stints at that company.
           - Provide the company name and the total tenure duration in years (rounded to two decimal places), e.g., "Amazon: 1.16 years".
           - Output the result as an array of strings, one per unique company, in the order they first appear in the candidate's experience.
           - Do not include any extra commentary or summary—just the array of company-wise total tenure durations. 

         8. Suggested Role:
         - Based on the candidate’s work experience, education, and technical skill set, suggest the most suitable job role they are likely to be both qualified for and interested in.
         - The suggested role should align with the candidate’s career progression, domain expertise, and strengths demonstrated in their resume.
         - Ensure the recommendation reflects realistic career advancement and industry relevance.

         9. Average Stability Across All Companies:
         - For each full-time experience listed in the candidate's resume, calculate the tenure in years using the StartDate and EndDate fields.
         - Ignore internships or training roles (e.g., titles containing "Intern" or "Trainee").
         - Store each company's tenure as part of the "StabilityAssessment" field in the format: "CompanyName: X.XX years".
         - Finally, compute the average stability across all valid companies and return it as a float rounded to two decimal places in the "AverageStability" field.
         - Return the result as a numeric value (in years), rounded to two decimal places.

         10. Company type match (Product/Service)
        - If all companies in candidate's experience are Product companies: "Product"
        - If all companies in candidate's experience are Service companies: "Service"
        - If all companies in candidate's experience are Banking companies: "Banking"
        - If all companies in candidate's experience are Product and Service companies: "Product/Service"
        - If candidate has mixed experience (both Product, Service, and Banking): "Product/Service/Banking"

        11.Business type match (B2B/B2C/combinations - consider partial matches for mixed models)
        - If all companies in candidate's experience are B2B companies: "B2B"
        - If all companies in candidate's experience are B2C companies: "B2C"
        - If all companies in candidate's experience are B2B and B2C companies: "B2B/B2C"
        - If candidate has mixed experience (both B2B and B2C): "B2B/B2C"

        12. Complex Work Experience 
          -Evaluate whether the candidate has worked at companies doing complex product development based on:
             - Ownership (builds own products),
             - Scale (handles high-traffic or low-latency systems),
             - Deep Tech (solves hard problems like AI, infra, or distributed systems),
             - Product-Led Culture, and
             - Reputation (engineering recognition or OSS presence).
            Mark "ComplexWorkExperience": true if at least one company meets these standards. Otherwise, return "false".

        IMPORTANT INSTRUCTIONS FOR COMPANY DETECTION:
        - Look for company names in work email addresses (e.g., @tcs.com suggests TCS)
        - Check LinkedIn URLs or profile mentions
        - Look for official company names, even if abbreviated (e.g., TCS = Tata Consultancy Services)
        - Identify subsidiaries and parent companies
        - If you recognize a company name, infer the company type and business type based on your knowledge
        
        COMPANY TYPE INFERENCE RULES:
        - TCS, Tata Consultancy Services, Infosys, Wipro, Accenture, Cognizant, IBM = Service companies
        - Amazon, Google, Microsoft, Apple, Meta, Netflix, Spotify = Product companies
        - Banks (JPMorgan, HDFC, ICICI), Consulting firms = Banking companies
        - Software products, E-commerce, SaaS platforms, Gaming companies = Product companies
        - Startups with apps/platforms = Product companies
        - IT Services, Consulting, Outsourcing = Service companies
        
        BUSINESS TYPE INFERENCE RULES:
        - IT Services companies (TCS, Infosys, Wipro) = services
        - E-commerce (Amazon, Flipkart) = B2C(product)
        - Enterprise software (Microsoft, Oracle) = B2B(product)
        - Social media (Meta, Twitter) = B2C(product)
        - Consumer products (Apple, Samsung) = B2B(product)
        - Banking and Financial Services = Banking(product)
        - Gaming companies = B2C/B2B(product)
        - SaaS platforms = B2B(product)
        
        FORMAT GUIDELINES:
        - Use B2B for pure Saas Companies 
        - Use B2B for pure enterprise Companies 
        - Use B2C for pure consumer services        
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
          "StabilityAssessment": ["string (company name and total tenure duration, e.g., 'Amazon: 1.16 years')", ...],
          "SuggestedRole": "string",
          "AverageStability": "string",
          "CompanyTypeMatch": "string (MUST be 'Product' if all CompanyAnalysis entries are Product type, 'Service' if all are Service type, 'Product/Service/Banking' only for mixed experience)",
          "BusinessTypeMatch": "string (explain the business type of the candidate)",
          "ComplexWorkExperience": "boolean ("true" if the candidate has worked at companies that meet key indicators of complex product development — such as owning their own IP, operating large-scale or low-latency systems, solving deep technical problems (e.g., AI/ML, distributed systems), having a product-led engineering culture, or being recognized in the tech ecosystem. "false" if none of the companies show evidence of such complexity.)"

        }
        
        EXAMPLES OF INFERENCE:
        - If resume mentions "worked at TCS" → CompanyName: "TCS", CompanyType: "Service"
        - If email is "john@amazon.com" → CompanyName: "Amazon", CompanyType: "Product", BusinessType: "B2C "
        - If mentions "Google India" → CompanyName: "Google", CompanyType: "Product", BusinessType: "B2C "
        - If mentions "JPMorgan Chase" → CompanyName: "JPMorgan Chase", CompanyType: "Banking", BusinessType: "B2B"
        - If mentions "Flipkart" → CompanyName: "Flipkart", CompanyType: "Product", BusinessType: "B2C "
        
        IMPORTANT INSTRUCTIONS:
        1. Extract dates EXACTLY as they appear in the resume without reformatting
        2. For Duration, maintain the exact format from the resume (e.g., "Jan 2020 - Mar 2022", "2019-Present")
        3. If a field is not present or cannot be determined, use null rather than making assumptions
        4. Be aggressive about finding company names from any source in the resume

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

