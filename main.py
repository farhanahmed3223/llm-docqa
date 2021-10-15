import openai
import os

openai.api_key = os.environ["OPENAI_API_KEY"]

text = open("doc.txt").read()
question = input("Question: ")

resp = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "Answer questions based on this text: " + text},
        {"role": "user", "content": question},
    ],
)
print(resp.choices[0].message.content)
