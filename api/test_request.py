import requests
from bs4 import BeautifulSoup

# Define the URL for your Flask app
url_generate = "http://127.0.0.1:5000/generate"
url_generate_answer = "http://127.0.0.1:5000/generate_answer"

# Test 1: Sending a prompt to the /generate endpoint
def test_generate():
    data = {
        "prompt": "Once upon a time, in a land far, far away"
    }
    response = requests.post(url_generate, data=data)
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
    response = requests.post(url_generate_answer, data=data)
    if response.status_code == 200:
        print("Test 2 (Generate Answer) passed")
        soup = BeautifulSoup(response.text, "html.parser")
        answer_tag = soup.find("h2", string="Generated Answer:")
        if answer_tag:
            answer_p = answer_tag.find_next("p")
            if answer_p:
                print("Generated Answer:", answer_p.string.split("Answer:")[-1].strip())
            else:
                print("Could not find the answer in the HTML response.")
        else:
            print("Could not locate 'Generated Answer:' section in the response.")
    else:
        print("Test 2 (Generate Answer) failed. Status code:", response.status_code)

# Run the tests
if __name__ == "__main__":
    #test_generate()
    test_generate_answer()
