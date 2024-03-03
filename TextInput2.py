import openai
openai.api_key = "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"

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

#Delimiters can be anything like: , """, < >, <tag> </tag>...

text = f"""I'm learning OpenAi API"""
prompt=f"""Tasks should be done and any required instructions...can u teach me how?{text}```"""
response=get_completion(prompt)
print(response)