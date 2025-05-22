import anthropic

# Replace 'your_anthropic_api_key' with your actual API key
api_key = 'your_anthropic_api_key'
client = anthropic.Client(api_key=api_key)

try:
    response = client.messages.create(
        model='claude-3-sonnet-20240229',
        messages=[{'role': 'user', 'content': 'Hello'}],
        max_tokens=1000
    )
    print("Access successful! Response:", response.content)
except Exception as e:
    print("Error:", e) 