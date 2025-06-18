from datetime import datetime, date
import re
from typing import Dict, List, Union
import json

def calculate_total_experience(resume_data: Dict) -> float:
    """
    Calculate total years of experience from resume data.
    
    Args:
        resume_data (Dict): Resume data containing experience information
        
    Returns:
        float: Total years of experience (rounded to 2 decimal places)
    """
    
    try:
        # Navigate to experience data
        if "steps" in resume_data and len(resume_data["steps"]) > 0:
            experience_list = resume_data["steps"][0].get("Experience", [])
        elif "Experience" in resume_data:
            experience_list = resume_data["Experience"]
        else:
            return 0.0
        
        total_months = 0
        
        for exp in experience_list:
            if "Duration" in exp:
                start_date = exp["Duration"].get("StartDate", "")
                end_date = exp["Duration"].get("EndDate", "")
                
                # Calculate months for this experience
                months = calculate_months_between_dates(start_date, end_date)
                total_months += months
        
        # Convert months to years and round to 2 decimal places
        total_years = round(total_months / 12, 2)
        return total_years
        
    except Exception as e:
        print(f"Error calculating experience: {e}")
        return 0.0


def calculate_months_between_dates(start_date: str, end_date: str) -> int:
    """
    Calculate months between two date strings.
    
    Args:
        start_date (str): Start date in various formats
        end_date (str): End date in various formats (can be "Present", "Current", etc.)
        
    Returns:
        int: Number of months between dates
    """
    
    try:
        # Parse start date
        start_dt = parse_date_string(start_date)
        if not start_dt:
            return 0
        
        # Parse end date
        if end_date.lower() in ["present", "current", "now", "till date"]:
            end_dt = datetime.now()
        else:
            end_dt = parse_date_string(end_date)
            if not end_dt:
                return 0
        
        # Calculate difference in months
        months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
        
        # Add partial month if end day is later than start day
        if end_dt.day >= start_dt.day:
            months += 1
        
        return max(0, months)  # Ensure non-negative
        
    except Exception as e:
        print(f"Error calculating months between {start_date} and {end_date}: {e}")
        return 0


def parse_date_string(date_str: str) -> Union[datetime, None]:
    """
    Parse various date string formats into datetime object.
    
    Args:
        date_str (str): Date string in various formats
        
    Returns:
        datetime or None: Parsed datetime object or None if parsing fails
    """
    
    if not date_str or not isinstance(date_str, str):
        return None
    
    # Clean the date string
    date_str = date_str.strip()
    
    # Common date formats to try
    date_formats = [
        "%B %Y",        # "February 2022", "July 2019"
        "%b %Y",        # "Feb 2022", "Jul 2019" 
        "%m/%Y",        # "02/2022"
        "%m-%Y",        # "02-2022"
        "%Y-%m",        # "2022-02"
        "%Y",           # "2022"
        "%B %d, %Y",    # "February 15, 2022"
        "%b %d, %Y",    # "Feb 15, 2022"
        "%d/%m/%Y",     # "15/02/2022"
        "%d-%m-%Y",     # "15-02-2022"
        "%Y-%m-%d",     # "2022-02-15"
    ]
    
    # Try each format
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # Try to extract year and month using regex
    try:
        # Look for month name and year
        month_year_pattern = r'(\w+)\s+(\d{4})'
        match = re.search(month_year_pattern, date_str)
        if match:
            month_str, year_str = match.groups()
            
            # Month name to number mapping
            months = {
                'january': 1, 'jan': 1,
                'february': 2, 'feb': 2,
                'march': 3, 'mar': 3,
                'april': 4, 'apr': 4,
                'may': 5,
                'june': 6, 'jun': 6,
                'july': 7, 'jul': 7,
                'august': 8, 'aug': 8,
                'september': 9, 'sep': 9, 'sept': 9,
                'october': 10, 'oct': 10,
                'november': 11, 'nov': 11,
                'december': 12, 'dec': 12
            }
            
            month_num = months.get(month_str.lower())
            if month_num:
                return datetime(int(year_str), month_num, 1)
        
        # Look for just year
        year_pattern = r'(\d{4})'
        match = re.search(year_pattern, date_str)
        if match:
            year = int(match.group(1))
            return datetime(year, 1, 1)  # Default to January 1st
            
    except Exception as e:
        print(f"Error parsing date with regex: {e}")
    
    return None


def add_total_experience_to_response(response_data: Dict) -> Dict:
    """
    Add TotalYearsOfExperience field to the response data.
    
    Args:
        response_data (Dict): The API response data
        
    Returns:
        Dict: Updated response data with TotalYearsOfExperience field
    """
    
    try:
        if "resume_data" in response_data:
            total_exp = calculate_total_experience(response_data["resume_data"])
            response_data["TotalYearsOfExperience"] = total_exp
        
        return response_data
        
    except Exception as e:
        print(f"Error adding total experience: {e}")
        return response_data


def get_experience_breakdown(resume_data: Dict) -> List[Dict]:
    """
    Get detailed breakdown of experience by company.
    
    Args:
        resume_data (Dict): Resume data containing experience information
        
    Returns:
        List[Dict]: List of experience details with calculated durations
    """
    
    try:
        # Navigate to experience data
        if "steps" in resume_data and len(resume_data["steps"]) > 0:
            experience_list = resume_data["steps"][0].get("Experience", [])
        elif "Experience" in resume_data:
            experience_list = resume_data["Experience"]
        else:
            return []
        
        breakdown = []
        
        for exp in experience_list:
            if "Duration" in exp:
                start_date = exp["Duration"].get("StartDate", "")
                end_date = exp["Duration"].get("EndDate", "")
                
                months = calculate_months_between_dates(start_date, end_date)
                years = round(months / 12, 2)
                
                breakdown.append({
                    "company": exp.get("CompanyName", "Unknown"),
                    "position": exp.get("Position", "Unknown"),
                    "start_date": start_date,
                    "end_date": end_date,
                    "duration_months": months,
                    "duration_years": years
                })
        
        return breakdown
        
    except Exception as e:
        print(f"Error getting experience breakdown: {e}")
        return []


# Example usage and testing
if __name__ == "__main__":
    # Test data similar to your response
    test_resume_data = {
        "steps": [
            {
                "Experience": [
                    {
                        "CompanyName": "Moneyview",
                        "Position": "Software Development Engineer",
                        "Duration": {
                            "StartDate": "Feb 2022",
                            "EndDate": "Present"
                        }
                    },
                    {
                        "CompanyName": "Amazon",
                        "Position": "Software Development Engineer",
                        "Duration": {
                            "StartDate": "July 2019",
                            "EndDate": "July 2020"
                        }
                    },
                    {
                        "CompanyName": "Amazon",
                        "Position": "Software Development Intern",
                        "Duration": {
                            "StartDate": "May 2018",
                            "EndDate": "July 2018"
                        }
                    }
                ]
            }
        ]
    }
    
    # Calculate total experience
    total_exp = calculate_total_experience(test_resume_data)
    print(f"Total Experience: {total_exp} years")
    
    # Get breakdown
    breakdown = get_experience_breakdown(test_resume_data)
    for exp in breakdown:
        print(f"{exp['company']}: {exp['duration_years']} years") 