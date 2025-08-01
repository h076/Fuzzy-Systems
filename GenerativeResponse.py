import string

import requests
import json

url = "http://localhost:11434/api/chat"

def cleanResponse(statement: string):
    prompt = ("I am going to provide you will a response that explains why a price was predicted, can you please " +
              "correct the grammar and only return the response. the response is : " + statement)

    data = {
        "model": "llama3",
        "messages": [
            {
                "role": "user",
                "content": prompt

            }
        ],
        "stream": False,
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()["message"]["content"]