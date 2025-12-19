# 🚀 Load Testing Guide - 10,000 Requests

## 📦 Quick Start

### Step 1: Install Requirements
```bash
pip install requests
```

### Step 2: Choose Your Script

#### **Option 1: Simple Load Test** (Recommended for quick tests)
```bash
python3 simple_load_test.py
```

#### **Option 2: Advanced Load Test** (Detailed statistics)
```bash
python3 load_test.py
```

---

## ⚙️ Configuration

### Simple Load Test (`simple_load_test.py`)

Edit these lines at the top of the file:

```python
SERVER_URL = "http://localhost:8000/api/v1/customer-app/business-categories/"
TOTAL_REQUESTS = 10000
CONCURRENT_THREADS = 100
```

**Change:**
- `SERVER_URL` → Your API endpoint
- `TOTAL_REQUESTS` → Number of requests (default: 10,000)
- `CONCURRENT_THREADS` → Concurrent requests (default: 100)

### Advanced Load Test (`load_test.py`)

Edit the configuration section:

```python
# Server configuration
BASE_URL = "http://localhost:8000"
API_ENDPOINT = "/api/v1/customer-app/business-categories/"

# Load test configuration
TOTAL_REQUESTS = 10000
CONCURRENT_WORKERS = 100
TIMEOUT = 30

# Request configuration
REQUEST_METHOD = "GET"  # GET, POST, PUT, DELETE
REQUEST_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_TOKEN"  # Add if needed
}
REQUEST_BODY = {}  # For POST/PUT requests
```

---

## 📋 Example Configurations

### Test GET Endpoint
```python
SERVER_URL = "http://localhost:8000/api/v1/customer-app/business-categories/"
REQUEST_METHOD = "GET"
```

### Test POST Endpoint
```python
BASE_URL = "http://localhost:8000"
API_ENDPOINT = "/api/v1/customer-app/discounts/verify/"
REQUEST_METHOD = "POST"
REQUEST_BODY = {
    "discount_code": "TEST123",
    "customer_id": 1
}
```

### Test with Authentication
```python
REQUEST_HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

### Test Multiple Endpoints
Create separate config files or run multiple times with different URLs.

---

## 🎯 Based on Your API (from screenshot)

### Customer App - Business Categories
```python
SERVER_URL = "http://localhost:8000/api/v1/customer-app/business-categories/"
REQUEST_METHOD = "GET"
```

### Customer App - Reviews
```python
SERVER_URL = "http://localhost:8000/api/v1/customer-app/customer/check-branch-review-status/"
REQUEST_METHOD = "GET"
```

### Customer App - Favourite Shops
```python
SERVER_URL = "http://localhost:8000/api/v1/customer-app/customer/check-favourite-shop/"
REQUEST_METHOD = "GET"
```

### Customer App - Discounts
```python
BASE_URL = "http://localhost:8000"
API_ENDPOINT = "/api/v1/customer-app/discounts/verify/"
REQUEST_METHOD = "POST"
REQUEST_BODY = {"code": "DISCOUNT123"}
```

---

## 🚀 Usage Examples

### Example 1: Quick Test
```bash
# Edit simple_load_test.py
# Change SERVER_URL to your endpoint
python3 simple_load_test.py
```

**Output:**
```
🚀 Sending 10000 requests to http://localhost:8000/api/...
⚡ Using 100 concurrent threads

✓ 100 requests sent | Success: 98 | Failed: 2
✓ 200 requests sent | Success: 196 | Failed: 4
...
✓ 10000 requests sent | Success: 9856 | Failed: 144

✅ DONE!
Total: 10000 | Success: 9856 | Failed: 144
Duration: 45.23s | RPS: 221.09
```

### Example 2: Detailed Statistics
```bash
python3 load_test.py
```

**Output:**
```
================================================================================
🚀 LOAD TESTING SCRIPT
================================================================================
Target URL: http://localhost:8000/api/v1/customer-app/business-categories/
Method: GET
Total Requests: 10,000
Concurrent Workers: 100
Timeout: 30s
================================================================================

⚠️  This will send 10,000 requests to your server. Continue? (yes/no): yes

🔥 Starting load test...

[██████████████████████████████████████████████████] 10000/10000 (100.0%) | 
RPS: 221.5 | ETA: 0.0s | Success: 9856 | Failed: 144

✅ Load test completed!

================================================================================
📊 LOAD TEST RESULTS
================================================================================
Total Requests:       10,000
Successful:           9,856
Failed:               144
Success Rate:         98.56%
Duration:             45.15s
Requests/Second:      221.49

Response Times (seconds):
  Min:                0.0123s
  Max:                2.3456s
  Average:            0.4521s
  Median:             0.4123s
  95th Percentile:    0.8765s
  99th Percentile:    1.2345s

Status Codes:
  200: 9,856
  500: 144

💾 Results saved to: load_test_results_20251215_224500.json
```

---

## ⚡ Performance Tuning

### Increase Concurrency (More aggressive)
```python
CONCURRENT_THREADS = 200  # or 500 for very aggressive testing
```

### Decrease Concurrency (Gentler on server)
```python
CONCURRENT_THREADS = 50  # or 10 for slow testing
```

### Adjust Total Requests
```python
TOTAL_REQUESTS = 1000    # Quick test
TOTAL_REQUESTS = 10000   # Standard test
TOTAL_REQUESTS = 100000  # Stress test
```

---

## 📊 Understanding Results

### Success Rate
- **>95%**: Excellent - Server handling load well
- **80-95%**: Good - Some issues under load
- **<80%**: Poor - Server struggling

### Requests Per Second (RPS)
- Higher is better
- Compare with your server's expected capacity

### Response Times
- **p95**: 95% of requests completed within this time
- **p99**: 99% of requests completed within this time
- Lower is better

---

## 🐛 Troubleshooting

### Issue: Connection Refused
**Solution:** Make sure your server is running
```bash
# Check if server is running
curl http://localhost:8000/api/v1/customer-app/business-categories/
```

### Issue: Too Many Failures
**Solutions:**
1. Reduce concurrent threads
2. Increase timeout
3. Check server logs
4. Verify endpoint URL

### Issue: Script Too Slow
**Solutions:**
1. Increase concurrent threads
2. Use simple_load_test.py instead
3. Run on a faster machine

### Issue: Server Crashes
**Solutions:**
1. Reduce concurrent threads
2. Reduce total requests
3. Add delays between requests

---

## 🎯 Test Scenarios

### Scenario 1: Baseline Test
```python
TOTAL_REQUESTS = 1000
CONCURRENT_THREADS = 10
```
**Purpose:** Establish baseline performance

### Scenario 2: Normal Load
```python
TOTAL_REQUESTS = 10000
CONCURRENT_THREADS = 100
```
**Purpose:** Simulate normal traffic

### Scenario 3: Stress Test
```python
TOTAL_REQUESTS = 50000
CONCURRENT_THREADS = 500
```
**Purpose:** Find breaking point

### Scenario 4: Spike Test
```python
TOTAL_REQUESTS = 10000
CONCURRENT_THREADS = 1000
```
**Purpose:** Test sudden traffic spike

---

## 📝 Best Practices

1. **Start Small**: Test with 100 requests first
2. **Monitor Server**: Watch CPU, memory, database connections
3. **Save Results**: Keep logs for comparison
4. **Test Different Endpoints**: Each may have different performance
5. **Test with Real Data**: Use realistic request bodies
6. **Test Authentication**: Include auth headers if needed
7. **Clean Up**: Clear test data after testing

---

## 🔧 Advanced Usage

### Test Multiple Endpoints Sequentially
```bash
# Test endpoint 1
python3 simple_load_test.py  # Configure for endpoint 1
# Wait for completion

# Test endpoint 2
python3 simple_load_test.py  # Configure for endpoint 2
```

### Save Results
```bash
python3 load_test.py > results.txt 2>&1
```

### Run in Background
```bash
nohup python3 load_test.py &
```

---

## 📞 Quick Reference

### Simple Test
```bash
# 1. Edit SERVER_URL in simple_load_test.py
# 2. Run:
python3 simple_load_test.py
```

### Advanced Test
```bash
# 1. Edit configuration in load_test.py
# 2. Run:
python3 load_test.py
```

### Change URL
```python
# In simple_load_test.py or load_test.py
SERVER_URL = "http://YOUR_SERVER/YOUR_ENDPOINT/"
```

---

**Ready to test! 🚀**

Choose your script and update the SERVER_URL, then run it!
