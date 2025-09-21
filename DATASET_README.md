# Processed Internship Dataset Documentation

## Overview
- **Processing Date**: 2025-09-21 16:57:28
- **Total Records**: 6,642
- **Total Features**: 30
- **Data Quality Score**: 83.9%

## Processing Steps
1. 1.1 - Data Loading and Initial Inspection
1. 1.2 - Missing Value Handling
1. 1.3 - Data Transformation and Feature Engineering
1. 1.4 - Data Validation and Quality Assurance
1. 1.5 - Data Export and Documentation

## Column Descriptions
| Column | Data Type | Null Count | Unique Values | Description |
|--------|-----------|------------|---------------|-------------|
| Internship Id | int64 | 0 | 6555 | Unique identifier for the internship |
| Role | object | 0 | 1681 | Position title of the internship |
| Company Name | object | 0 | 4757 | Name of the company offering the internship |
| Location | object | 0 | 990 | Original location information |
| Duration | int64 | 0 | 9 | Duration of internship in months (numeric) |
| Stipend | object | 0 | 392 | Original stipend information as string |
| Intern Type | object | 0 | 8 | Type of internship (e.g., Full-time, Part-time) |
| Skills | object | 0 | List data | List of required skills for the internship |
| Perks | object | 0 | List data | List of perks offered with the internship |
| Hiring Since | object | 2 | 123 | How long the company has been hiring |
| Opportunity Date | object | 1 | 358 | Date when opportunity was posted |
| Opening | int64 | 0 | 48 | Number of open positions |
| Hired Candidate | object | 3154 | 215 | Information about hired candidates |
| Number of Applications | float64 | 0 | 1 | Count of applications received |
| Website Link | object | 2330 | 3267 | URL to apply for the internship |
| Stipend Amount | float64 | 32 | 78 | Extracted numeric stipend amount |
| Stipend Currency | object | 0 | 8 | Currency of the stipend amount |
| Has Skills | bool | 0 | 2 | Boolean indicating if skills are specified |
| Has Perks | bool | 0 | 2 | Boolean indicating if perks are specified |
| Stipend Specified | bool | 0 | 2 | Boolean indicating if stipend is specified |
| City | object | 0 | 207 | Extracted city from location |
| Country | object | 0 | 75 | Extracted country from location |
| Duration Category | object | 0 | 4 | Categorized duration (Short-term, Medium-term, Long-term, Extended) |
| Application Volume | object | 0 | 1 | Categorized application volume |
| Posting Date | datetime64[ns] | 6642 | 0 | Date when internship was posted |
| Posting Month | float64 | 6642 | 0 | Month when internship was posted |
| Posting Year | float64 | 6642 | 0 | Year when internship was posted |
| Days Since Posted | float64 | 6642 | 0 | Number of days since internship was posted |
| Number of Skills | int64 | 0 | 21 | Count of skills required |
| Number of Perks | int64 | 0 | 8 | Count of perks offered |

## Quality Metrics
- **Completeness Rate**: 83.90%
- **Null Cells**: 32087
- **Total Cells**: 199260

## Validation Results Summary
- **Passed Checks**: 10/13 (76.9%)

## Files Generated
- `processed_internship_data.csv`: Main dataset in CSV format
- `processed_internship_data.json`: Dataset in JSON format for web applications
- `dataset_documentation.json`: Comprehensive documentation in JSON format
- `data_validation_report.csv`: Detailed validation results
