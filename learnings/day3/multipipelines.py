from transformers import pipeline
sa=pipeline("sentiment-analysis")
print(sa("This food quality is excellent!"))

ner=pipeline("ner",aggregation_strategy="simple")
print(ner("Infosys is based in bengaluru,Karnatka."))

qa=pipeline("question-answering")
context="Python was created by Guido van tossum in 1991.It is widely used in AI"
print(qa(question="who found python",context=context))

summ=pipeline("summarization")
long_text="Generative Ai refers to AI systems that can generate new content..."
print(summ(long_text,max_length=60,min_length=20))