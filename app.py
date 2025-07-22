import os
import tempfile
import shutil
import uuid
import json
import requests
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import Dict, Any, Optional

from openai_batch_resume_agents import analyze_resume
# from parallel_resume_agents import analyze_resume 
# from gemini_parallel_resume_agents import analyze_resume
# from resume_agent import analyze_resume 
from jd_agent import analyze_jd
from analyze import analyze_resume_and_jd
from text_extractor import extract_text_from_file
from cost_calculator import calculations_cost
from experience_calculator import calculate_total_experience
# Removed resume_enricher imports as web search is now integrated directly in resume_agent

# Create FastAPI app
app = FastAPI(title="Resume Parser API", description="API for parsing resumes and job descriptions")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class JobDescription(BaseModel):
    jd: str

class MatchRequest(BaseModel):
    resume_id: str
    jd_id: str

# Removed EnrichRequest class as enrichment is now integrated directly in resume_agent

@app.post("/upload-resume/", response_model=Dict[str, Any])
async def upload_resume(
    resume_file: UploadFile = File(...)
):
    """Upload and process a resume file (PDF or DOCX)"""
    # Check file extension
    file_extension = os.path.splitext(resume_file.filename)[1].lower()
    if file_extension not in ['.pdf', '.docx', '.txt']:
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a PDF, DOCX, or TXT file.")
    
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
    
    try:
        # Write the uploaded file to the temporary file
        shutil.copyfileobj(resume_file.file, temp_file)
        temp_file.close()
        
        # Extract text from the file
        import time
        start = time.time()
        extracted_text = extract_text_from_file(temp_file.name)
        end = time.time()
        print(f"Time taken: {end - start} seconds")
        
        # Process the resume with extracted text (includes automatic web search enrichment for null fields)
        result, total_tokens = await analyze_resume(extracted_text)
        cost_info = calculations_cost(total_tokens)
        
        # Parse the JSON result
        resume_data_raw = json.loads(result)
        
        # Flatten the resume data (remove "steps" wrapper)
        if "steps" in resume_data_raw and len(resume_data_raw["steps"]) > 0:
            resume_data = resume_data_raw["steps"][0]
        else:
            resume_data = resume_data_raw
        
        # Calculate total years of experience
        total_experience = calculate_total_experience({"steps": [resume_data]} if "steps" not in resume_data_raw else resume_data_raw)
        
        # Generate a unique ID for this resume
        resume_id = str(uuid.uuid4())
        
        # Get current timestamp
        upload_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create response structure matching desired format
        response = {
            "status": "success",
            "resume_id": resume_id,
            "resume_data": resume_data,
            "TotalYearsOfExperience": total_experience,
            "upload_date": upload_date,
            "usage": {
                "tokens": total_tokens,
                "cost": round(cost_info.get("estimated_total_cost_usd", 0), 5)
            }
        }
        
        return response
    
    except Exception as e:
        print(f"Error processing resume: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing resume: {str(e)}")
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


@app.post("/upload-jd/", response_model=Dict[str, Any])
async def upload_jd(jd_data: JobDescription):
    """Process a job description provided as text"""
    try:
        # Process the job description
        result, total_tokens = analyze_jd(jd_data.jd)
        cost_info = calculations_cost(total_tokens)
        
        # Parse the JSON result
        jd_data_parsed = json.loads(result)
        
        # Flatten the JD data (remove "steps" wrapper if present)
        if "steps" in jd_data_parsed and len(jd_data_parsed["steps"]) > 0:
            jd_data_clean = jd_data_parsed["steps"][0]
        else:
            jd_data_clean = jd_data_parsed
        
        # Generate a unique ID for this job description
        jd_id = str(uuid.uuid4())
        
        # Get current timestamp
        upload_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create response structure matching the resume format
        response = {
            "status": "success",
            "jd_id": jd_id,
            "jd_data": jd_data_clean,
            "upload_date": upload_date,
            "usage": {
                "tokens": total_tokens,
                "cost": round(cost_info.get("estimated_total_cost_usd", 0), 5)
            }
        }
        
        return response
    
    except Exception as e:
        print(f"Error processing job description: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing job description: {str(e)}")

@app.post("/analyze-match/", response_model=Dict[str, Any])
async def analyze_match(match_request: MatchRequest):
    """Analyze how well the resume matches the job description"""
    
    resume_id = match_request.resume_id
    jd_id = match_request.jd_id
    
    try:
        # Fetch resume data from external API
        resume_response = requests.post(
            "https://nodeapi.hiringeye.ai/api/v1/other/search-resume",
            json={"resume_id": resume_id}
        )
        if not resume_response.ok:
            raise HTTPException(status_code=404, detail=f"Resume with ID {resume_id} not found in external API")
        
        resume_data = resume_response.json()
        if not resume_data.get("status"):
            raise HTTPException(status_code=404, detail=f"Resume with ID {resume_id} not found: {resume_data.get('message')}")
        
        # Extract only the resume data we need for analysis
        resume_info = resume_data.get("data", {})
        
        # Fetch JD data from external API
        jd_response = requests.post(
            "https://nodeapi.hiringeye.ai/api/v1/other/search-jd",
            json={"jd_id": jd_id}
        )
        if not jd_response.ok:
            raise HTTPException(status_code=404, detail=f"Job description with ID {jd_id} not found in external API")
        
        jd_data = jd_response.json()
        if not jd_data.get("status"):
            raise HTTPException(status_code=404, detail=f"Job description with ID {jd_id} not found: {jd_data.get('message')}")
        
        # Extract only the JD data we need for analysis
        jd_info = jd_data.get("data", {})
        
        # Clean up the resume and JD data by removing unwanted fields
        cleaned_resume = {
            "SuggestedRole": resume_info.get("suggested_role"),
            "CandidateFullName": resume_info.get("candidate_full_name"),
            "EmailAddress": resume_info.get("email_address"),
            "PhoneNumber": resume_info.get("phone_number"),
            "Skills": resume_info.get("skills", []),
            "Experience": resume_info.get("experience", []),
            "Education": resume_info.get("education_details", []),
            "StabilityAssessment": resume_info.get("overall_stability_assessment"),
            "TotalYearsOfExperience": resume_info.get("total_years_of_experience", 0.0),
            "resume_file": resume_info.get("resume_file"),
            "upload_date": resume_info.get("upload_date")
        }
        
        cleaned_jd = {
            "CompanyName": jd_info.get("company_name"),
            "JobTitle": jd_info.get("job_title"),
            "RequiredSkills": jd_info.get("required_skills", {"technical": [], "soft": []}),
            "YearsOfExperienceRequired": jd_info.get("years_of_experience_required"),
            "EducationRequirements": jd_info.get("education_requirements"),
            "CompanyTypePreference": jd_info.get("company_type_preference"),
            "BusinessTypePreference": jd_info.get("business_type_preference"),
            "PreferredStability": jd_info.get("preferred_stability"),
            "OtherImportantRequirements": jd_info.get("other_important_requirements", []),
            "jd_file": jd_info.get("jd_file"),
            "upload_date": jd_info.get("upload_date")
        }
        
        # Combine resume and JD data for analysis
        combined_input = f"""
        Resume data:
        {json.dumps(cleaned_resume, indent=2)}
        
        Job Description data:
        {json.dumps(cleaned_jd, indent=2)}
        """
        # print(combined_input)
        # Analyze the match using existing function
        result, total_tokens = analyze_resume_and_jd(combined_input)
        cost_info = calculations_cost(total_tokens)
        
        # Parse the JSON result
        analysis_data = json.loads(result)
        
        # Flatten the analysis data (remove "steps" wrapper if present)
        if "steps" in analysis_data and len(analysis_data["steps"]) > 0:
            analysis_clean = analysis_data["steps"][0]
        else:
            analysis_clean = analysis_data
        
        # Get current timestamp
        analysis_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create response structure
        response = {
            "status": "success",
            "resume_id": resume_id,
            "jd_id": jd_id,
            "analysis": analysis_clean,
            "analysis_date": analysis_date,
            "usage": {
                "tokens": total_tokens,
                "cost": round(cost_info.get("estimated_total_cost_usd", 0), 5)
            }
        }
        
        return response
    
    except Exception as e:
        print(f"Error analyzing match: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing match: {str(e)}")
    


if __name__ == "__main__":
    uvicorn.run("app:app", host="121.0.0.1", port=8545, reload=True)