# Load Test Configuration for Salon Stylz API

## Overview
This load test is configured to test the **Branch Creation API** endpoint on the Salon Stylz platform.

## Endpoint Details
- **URL**: `https://salon.stylz.me/api/v1/shop/branches/create/`
- **Method**: POST
- **Content-Type**: application/json

## Test Configuration
- **Total Requests**: 10,000
- **Concurrent Workers**: 100 threads
- **Timeout**: 30 seconds per request

## Request Body Schema
```json
{
  "branch_name_en": "string",
  "branch_name_ar": "string",
  "notification_en": "string",
  "notification_ar": "string",
  "lat": "31.5204",
  "lng": "74.3587",
  "country_code": "+92",
  "city": "string",
  "address": "string",
  "phone": "03001234567",
  "business_category": 0
}
```

## Important Notes

### ⚠️ Authentication Required
The API endpoint likely requires authentication. Before running the load test:

1. **Check if authentication is needed** - The endpoint may require a Bearer token or API key
2. **Update the headers** in `load_test.py` if authentication is required:
   ```python
   REQUEST_HEADERS = {
       "Content-Type": "application/json",
       "User-Agent": "LoadTester/1.0",
       "Authorization": "Bearer YOUR_ACTUAL_TOKEN_HERE"
   }
   ```

### ⚠️ Data Considerations
- This test will attempt to **CREATE 10,000 branches** in the database
- **This will likely cause issues** unless:
  - The API has duplicate detection
  - You have permission to create test data
  - The database can handle this volume

### Recommendations Before Running

1. **Start with a smaller test**:
   ```python
   TOTAL_REQUESTS = 10  # Start with just 10 requests
   CONCURRENT_WORKERS = 2  # Use fewer workers
   ```

2. **Verify the request body**:
   - Update the sample data with realistic values
   - Ensure `business_category` ID exists in the system
   - Use unique values for each request if needed

3. **Check with the API owner**:
   - Confirm you have permission to run load tests
   - Ask about rate limiting
   - Verify if there's a test/staging environment

## How to Run

### Step 1: Install Dependencies
```bash
pip install requests
```

### Step 2: Configure the Script
Edit `load_test.py` and update:
- Authentication headers (if required)
- Request body with valid data
- Reduce TOTAL_REQUESTS for initial testing

### Step 3: Run the Test
```bash
python3 load_test.py
```

### Step 4: Review Results
The script will:
- Show real-time progress
- Display detailed statistics after completion
- Save results to a JSON file: `load_test_results_YYYYMMDD_HHMMSS.json`

## Metrics Collected
- Total requests sent
- Success/failure rate
- Response time statistics (min, max, avg, median, p95, p99)
- Status code distribution
- Requests per second (RPS)
- Error details

## Example Output
```
🚀 LOAD TESTING SCRIPT
================================================================================
Target URL: https://salon.stylz.me/api/v1/shop/branches/create/
Method: POST
Total Requests: 10,000
Concurrent Workers: 100
Timeout: 30s
================================================================================

[████████████████████████████████████████] 10000/10000 (100.0%) | 
RPS: 250.5 | ETA: 0.0s | Success: 8500 | Failed: 1500

📊 LOAD TEST RESULTS
================================================================================
Total Requests:       10,000
Successful:           8,500
Failed:               1,500
Success Rate:         85.00%
Duration:             39.92s
Requests/Second:      250.50

Response Times (seconds):
  Min:                0.0234s
  Max:                2.4567s
  Average:            0.3456s
  Median:             0.2987s
  95th Percentile:    0.8765s
  99th Percentile:    1.2345s
```

## Troubleshooting

### Common Issues

1. **401 Unauthorized**
   - Add authentication token to headers

2. **429 Too Many Requests**
   - Reduce CONCURRENT_WORKERS
   - Add delays between requests

3. **422 Unprocessable Entity**
   - Check request body format
   - Verify required fields are present

4. **500 Internal Server Error**
   - The server may be overwhelmed
   - Reduce load or check with API owner

## Safety Checklist
- [ ] I have permission to run load tests on this API
- [ ] I have configured authentication properly
- [ ] I have tested with a small number of requests first
- [ ] I understand this will create data in the system
- [ ] I have verified the request body contains valid data
- [ ] I have checked for rate limiting policies
