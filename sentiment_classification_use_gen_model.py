from transformers import pipeline

# Load the model
generator = pipeline("text-generation", model="microsoft/Phi-3-mini-4k-instruct")

# Input text and sentiment classification
prompt = "Text: 'I love this product so much!'\nClassification: "
output = generator(prompt, max_new_tokens=10, do_sample=False)

print(output[0]["generated_text"])
