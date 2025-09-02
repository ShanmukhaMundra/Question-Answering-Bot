import nltk
import json
print(f"Hello! I'm Atum, a question answering bot who knows answers to all questions from the 'Jeopardy!' game.")
print('Ask me something!')
user_input = input(">")
print("Let's play!")
with open('data/atum.json', 'r') as f:
    data = json.load(f)
def tokenize(user_input):
    text = user_input.lower()
    replacement = ['?', '!', '.', ',', ';', '&', '*', '_', '+', '=', '`']
    for mark in replacement:
        text = text.replace(mark, '')
    tokens = nltk.word_tokenize(text)
    return tokens
print(tokenize(user_input))