import requests

# Define the URL for your Flask app
url_generate = "http://127.0.0.1:5000/generate"
url_generate_answer = "http://127.0.0.1:5000/generate_answer"

# Test 1: Sending a prompt to the /generate endpoint
def test_generate():
    data = {
        "prompt": "Once upon a time, in a land far, far away"
    }
    response = requests.post(url_generate, json=data)
    if response.status_code == 200:
        print("Test 1 (Generate Text) passed")
        print("Generated Response:", response.json()["response"])
    else:
        print("Test 1 (Generate Text) failed. Status code:", response.status_code)

# Test 2: Sending context and question to the /generate_answer endpoint
def test_generate_answer():
    data = {
        "context": "The capital of France is Paris.",
        "question": "What is the capital of France?"
    }
    response = requests.post(url_generate_answer, json=data)
    if response.status_code == 200:
        print("Test 2 (Generate Answer) passed")
        print("Generated Answer:", response.json()["answer"])
    else:
        print("Test 2 (Generate Answer) failed. Status code:", response.status_code)

# Run the tests
if __name__ == "__main__":
    #test_generate()
    test_generate_answer()
