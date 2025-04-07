import requests
from bs4 import BeautifulSoup

url_generate_answer = "http://127.0.0.1:5000/generate_answer"

def test_generate_answer():
    data = {
        "model": "distilgpt2",
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


if __name__ == "__main__":
    test_generate_answer()
