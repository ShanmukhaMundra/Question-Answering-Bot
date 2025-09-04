import json
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

user_input = input(">")

with open('data/atum.json', 'r') as f:
    data = json.load(f)

def preprocess(text):
    text = text.lower()
    for char in ['?', '!', '.', ',']:
        text = text.replace(char, '')
    return word_tokenize(text)

tagged_data = []
for i, item in enumerate(data):
    question = item.get("question", "").strip()
    if question:
        tokens = preprocess(question)
        if tokens:
            tagged_data.append(TaggedDocument(words=tokens, tags=[str(i)]))

model = Doc2Vec(vector_size= 100, window=5, min_count=2, workers=4, epochs=40)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

user_tokens = preprocess(user_input)
if not user_tokens:
    print("Sorry, I couldn't understand your question.")
else:
    inferred_vector = model.infer_vector(user_tokens)
    most_similar = model.dv.most_similar([inferred_vector], topn=1)[0]

    closest_index, similarity_score = most_similar
    similarity_percent = round(similarity_score * 100, 2)

    print(f"I know this question: its number is {closest_index}. I'm {similarity_percent}% sure of this.")
