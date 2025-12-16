#!/usr/bin/env python3
"""
Simple Load Test - Quick 10,000 requests
"""

import requests
from concurrent.futures import ThreadPoolExecutor
import time

# ============================================================================
# CONFIGURATION - EDIT THESE
# ============================================================================

SERVER_URL = "http://localhost:8000/api/v1/customer-app/business-categories/"
TOTAL_REQUESTS = 10000
CONCURRENT_THREADS = 100

# ============================================================================

success_count = 0
fail_count = 0

def send_request(i):
    global success_count, fail_count
    try:
        response = requests.get(SERVER_URL, timeout=30)
        if response.status_code == 200:
            success_count += 1
        else:
            fail_count += 1
        if i % 100 == 0:
            print(f"✓ {i} requests sent | Success: {success_count} | Failed: {fail_count}")
    except Exception as e:
        fail_count += 1

print(f"🚀 Sending {TOTAL_REQUESTS} requests to {SERVER_URL}")
print(f"⚡ Using {CONCURRENT_THREADS} concurrent threads\n")

start_time = time.time()

with ThreadPoolExecutor(max_workers=CONCURRENT_THREADS) as executor:
    executor.map(send_request, range(TOTAL_REQUESTS))

duration = time.time() - start_time

print(f"\n✅ DONE!")
print(f"Total: {TOTAL_REQUESTS} | Success: {success_count} | Failed: {fail_count}")
print(f"Duration: {duration:.2f}s | RPS: {TOTAL_REQUESTS/duration:.2f}")
