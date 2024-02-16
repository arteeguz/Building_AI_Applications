from transformers import pipeline 
from transformers import AutoTokenizer, AutoModelForSequenceClassification

classifier = pipeline("sentiment-analysis")

res = classifier("I've been waiting for a HuggingFace course my whole life.")

print(res)

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer=AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

res = classifier("I've been waiting for a HuggingFace course my whole life.")

print(res)

seqience = "Using a Transformer network is simple"
res = tokenizer(seqience)
print(res)
tokens=tokenizer.tokenize(seqience)
print(tokens)
ids=tokenizer.convert_tokens_to_ids(tokens)
print(ids)
decoded_string = tokenizer.decode(ids)
print(decoded_string)