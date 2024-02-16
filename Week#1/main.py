from transformers import pipeline

classifier = pipeline("sentiment-analysis")

res = classifier("I've been waiting for a HugginFace course my whole life.")

print(res)

generator = pipeline("text-generation", model="distilgpt2")

res = generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
print(res)

classifier=pipeline("zero-shot-classification")

res = classifier(
    "This is a course about Python list comprehension",
    candidate_labels=["education", "politics", "business"],
)

print(res)