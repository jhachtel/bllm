#!/usr/bin/python3

import re
import urllib.request

url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
# print("Total number of characters: ", len(raw_text))
# print(raw_text[:99])

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
# print(len(preprocessed))

all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

vocab = {token:integer for integer,token in enumerate(all_words)}

# for i, item in enumerate(vocab.items()):
#     print(item)
#     if i >= 50:
#         break


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


tokenizer = SimpleTokenizerV1(vocab)
text = "It's the last he painted , you know, Mrs. Gisburn said with pardonable pride."
ids = tokenizer.encode(text)
# print(ids)

# print(tokenizer.decode(ids))

# text = "Hello, do you like tea?"
# print(tokenizer.encode(text))

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "|<unk>|"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}

print(len(vocab.items()))

for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)
