import requests
import time
import os

API_URL = "http://localhost:8000"

def test_upload_txt():
    print("Testing TXT upload...")
    with open("sample_test.txt", "w") as f:
        f.write("The quick brown fox jumps over the lazy dog. This is a test document for RAG system verification.")
    
    files = {"file": ("sample_test.txt", open("sample_test.txt", "rb"), "text/plain")}
    response = requests.post(f"{API_URL}/upload", files=files)
    print(f"Response: {response.status_code}, {response.json()}")
    os.remove("sample_test.txt")
    return response.status_code == 200

def test_ask():
    print("\nTesting Query...")
    payload = {"question": "What does the fox do?", "k": 2}
    response = requests.post(f"{API_URL}/ask", json=payload)
    print(f"Response: {response.status_code}, {response.json()}")
    return response.status_code == 200

def test_rate_limiting():
    print("\nTesting Rate Limiting...")
    payload = {"question": "Test", "k": 1}
    for i in range(25):
        response = requests.post(f"{API_URL}/ask", json=payload)
        if response.status_code == 429:
            print(f"Rate limit hit at request {i+1}")
            return True
    print("Rate limit not hit (might need lower limit for test)")
    return False

if __name__ == "__main__":
    # Note: These tests assume the server is running on localhost:8000
    # In a real scenario, we would start/stop the server in the script.
    print("Note: Ensure the app is running (streamlit run app.py) before running this script.")
    try:
        if test_upload_txt():
            time.sleep(2) # Wait for background processing
            test_ask()
            test_rate_limiting()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Please run the app first.")
