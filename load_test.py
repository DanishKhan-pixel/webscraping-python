#!/usr/bin/env python3
"""
Load Testing Script - Send 10,000 requests to your server
Supports concurrent requests with detailed statistics
"""

import requests
import time
import threading
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from datetime import datetime
import statistics

# ============================================================================
# CONFIGURATION
# ============================================================================

# Server configuration
BASE_URL = "http://localhost:8000"  # Change to your server URL
API_ENDPOINT = "/api/v1/customer-app/business-categories/"  # Change to your endpoint

# Load test configuration
TOTAL_REQUESTS = 10000
CONCURRENT_WORKERS = 100  # Number of concurrent threads
TIMEOUT = 30  # Request timeout in seconds

# Request configuration
REQUEST_METHOD = "GET"  # GET, POST, PUT, DELETE
REQUEST_HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "LoadTester/1.0"
}
REQUEST_BODY = {}  # For POST/PUT requests

# ============================================================================
# STATISTICS TRACKING
# ============================================================================

class LoadTestStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.response_times = []
        self.status_codes = defaultdict(int)
        self.errors = defaultdict(int)
        self.start_time = None
        self.end_time = None
    
    def record_request(self, success, response_time, status_code=None, error=None):
        with self.lock:
            self.total_requests += 1
            if success:
                self.successful_requests += 1
                self.response_times.append(response_time)
                if status_code:
                    self.status_codes[status_code] += 1
            else:
                self.failed_requests += 1
                if error:
                    self.errors[str(error)] += 1
    
    def get_summary(self):
        with self.lock:
            if not self.response_times:
                return None
            
            duration = (self.end_time - self.start_time) if self.end_time else 0
            
            return {
                "total_requests": self.total_requests,
                "successful": self.successful_requests,
                "failed": self.failed_requests,
                "success_rate": (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0,
                "duration_seconds": duration,
                "requests_per_second": self.total_requests / duration if duration > 0 else 0,
                "response_times": {
                    "min": min(self.response_times),
                    "max": max(self.response_times),
                    "avg": statistics.mean(self.response_times),
                    "median": statistics.median(self.response_times),
                    "p95": statistics.quantiles(self.response_times, n=20)[18] if len(self.response_times) > 20 else max(self.response_times),
                    "p99": statistics.quantiles(self.response_times, n=100)[98] if len(self.response_times) > 100 else max(self.response_times),
                },
                "status_codes": dict(self.status_codes),
                "errors": dict(self.errors)
            }

# ============================================================================
# REQUEST FUNCTIONS
# ============================================================================

stats = LoadTestStats()

def make_request(request_id):
    """Make a single HTTP request and record statistics."""
    url = f"{BASE_URL}{API_ENDPOINT}"
    
    start_time = time.time()
    
    try:
        if REQUEST_METHOD == "GET":
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=TIMEOUT)
        elif REQUEST_METHOD == "POST":
            response = requests.post(url, headers=REQUEST_HEADERS, json=REQUEST_BODY, timeout=TIMEOUT)
        elif REQUEST_METHOD == "PUT":
            response = requests.put(url, headers=REQUEST_HEADERS, json=REQUEST_BODY, timeout=TIMEOUT)
        elif REQUEST_METHOD == "DELETE":
            response = requests.delete(url, headers=REQUEST_HEADERS, timeout=TIMEOUT)
        else:
            raise ValueError(f"Unsupported method: {REQUEST_METHOD}")
        
        response_time = time.time() - start_time
        
        # Consider 2xx and 3xx as success
        success = 200 <= response.status_code < 400
        
        stats.record_request(
            success=success,
            response_time=response_time,
            status_code=response.status_code
        )
        
        return {
            "id": request_id,
            "success": success,
            "status_code": response.status_code,
            "response_time": response_time
        }
        
    except Exception as e:
        response_time = time.time() - start_time
        stats.record_request(
            success=False,
            response_time=response_time,
            error=str(e)
        )
        
        return {
            "id": request_id,
            "success": False,
            "error": str(e),
            "response_time": response_time
        }

# ============================================================================
# PROGRESS DISPLAY
# ============================================================================

def print_progress(current, total, start_time):
    """Print progress bar and statistics."""
    elapsed = time.time() - start_time
    percentage = (current / total) * 100
    bar_length = 50
    filled = int(bar_length * current / total)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    rps = current / elapsed if elapsed > 0 else 0
    eta = (total - current) / rps if rps > 0 else 0
    
    print(f"\r[{bar}] {current}/{total} ({percentage:.1f}%) | "
          f"RPS: {rps:.1f} | ETA: {eta:.1f}s | "
          f"Success: {stats.successful_requests} | Failed: {stats.failed_requests}", 
          end='', flush=True)

# ============================================================================
# MAIN LOAD TEST
# ============================================================================

def run_load_test():
    """Execute the load test with concurrent requests."""
    
    print("="*80)
    print("🚀 LOAD TESTING SCRIPT")
    print("="*80)
    print(f"Target URL: {BASE_URL}{API_ENDPOINT}")
    print(f"Method: {REQUEST_METHOD}")
    print(f"Total Requests: {TOTAL_REQUESTS:,}")
    print(f"Concurrent Workers: {CONCURRENT_WORKERS}")
    print(f"Timeout: {TIMEOUT}s")
    print("="*80)
    print()
    
    # Confirm before starting
    response = input("⚠️  This will send 10,000 requests to your server. Continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("❌ Load test cancelled.")
        return
    
    print("\n🔥 Starting load test...\n")
    
    stats.start_time = time.time()
    
    # Execute requests concurrently
    with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
        futures = [executor.submit(make_request, i) for i in range(TOTAL_REQUESTS)]
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 100 == 0 or completed == TOTAL_REQUESTS:
                print_progress(completed, TOTAL_REQUESTS, stats.start_time)
    
    stats.end_time = time.time()
    
    print("\n\n✅ Load test completed!\n")
    
    # Print summary
    summary = stats.get_summary()
    if summary:
        print("="*80)
        print("📊 LOAD TEST RESULTS")
        print("="*80)
        print(f"Total Requests:       {summary['total_requests']:,}")
        print(f"Successful:           {summary['successful']:,}")
        print(f"Failed:               {summary['failed']:,}")
        print(f"Success Rate:         {summary['success_rate']:.2f}%")
        print(f"Duration:             {summary['duration_seconds']:.2f}s")
        print(f"Requests/Second:      {summary['requests_per_second']:.2f}")
        print()
        print("Response Times (seconds):")
        print(f"  Min:                {summary['response_times']['min']:.4f}s")
        print(f"  Max:                {summary['response_times']['max']:.4f}s")
        print(f"  Average:            {summary['response_times']['avg']:.4f}s")
        print(f"  Median:             {summary['response_times']['median']:.4f}s")
        print(f"  95th Percentile:    {summary['response_times']['p95']:.4f}s")
        print(f"  99th Percentile:    {summary['response_times']['p99']:.4f}s")
        print()
        print("Status Codes:")
        for code, count in sorted(summary['status_codes'].items()):
            print(f"  {code}: {count:,}")
        
        if summary['errors']:
            print()
            print("Errors:")
            for error, count in sorted(summary['errors'].items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {error[:60]}: {count:,}")
        
        print("="*80)
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"load_test_results_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n💾 Results saved to: {filename}")

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        run_load_test()
    except KeyboardInterrupt:
        print("\n\n⚠️  Load test interrupted by user.")
        stats.end_time = time.time()
        summary = stats.get_summary()
        if summary:
            print(f"\n📊 Partial results: {summary['total_requests']} requests completed")
    except Exception as e:
        print(f"\n❌ Error: {e}")
