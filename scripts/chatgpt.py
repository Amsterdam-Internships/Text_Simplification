import requests
import time

simplified_sentences = []
source_file = "NMT-Data/Eval_Medical_Dutch_C_Dutch_S/NL_test_org"


# Set up API endpoint
api_url = 'https://api.openai.com/v1/chat/completions'

# Set up request headers
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer YOUR_API_KEY'  # Replace YOUR_API_KEY with your actual API key
}

data = {
    'model': 'gpt-3.5-turbo',
    'messages': [{'role': 'user', 'content': 'Placeholder'}]
}

error_count = 0

with open(source_file, 'r') as file:
    for line in file:
        sentence = line.strip()
        data['messages'][0]['content'] = f"Can you simplify the following sentence in dutch? {sentence}"
        response = requests.post(api_url, headers=headers, json=data)
        # Process the response
        if response.status_code == 200:
            result = response.json()
            content_text = result['choices'][0]['message']['content']
            simplified_sentences.append(content_text)
        else:
            error_count += 1
            content_text = sentence
            simplified_sentences.append(content_text)
            time.sleep(5)
            print('Request failed with status code:', response.status_code)

def write_list_to_file(lst, filename):
    with open(filename, 'w') as file:
        for item in lst:
            file.write(str(item) + '\n')


output_file = 'NMT-Data/GPT/model_output/med_pred_gpt'
write_list_to_file(simplified_sentences, output_file)
print(f"During simplification, encountered {error_count} errors that were replaced with the original sentence.")