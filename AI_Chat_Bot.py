import requests

def nvidia_api_call(query, api_key, invoke_url, fetch_url_format):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }

    payload = {
        "messages": [
            {
                "content": query,
                "role": "user"
            }
        ],
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 1024,
        "stream": False
    }

    session = requests.Session()
    response = session.post(invoke_url, headers=headers, json=payload)

    while response.status_code == 202:
        request_id = response.headers.get("NVCF-REQID")
        fetch_url = fetch_url_format + request_id
        response = session.get(fetch_url, headers=headers)

    response.raise_for_status()
    response_body = response.json()
    return response_body['choices'][0]['message']['content']

api_key = "API KEY"
invoke_url = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/0e349b44-440a-44e1-93e9-abe8dcb27158"
fetch_url_format = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/"

# recommendations = nvidia_api_call(prompt, api_key, invoke_url, fetch_url_format)
# print(recommendations)