# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:31:02 2024

@author: jeeva
"""



from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import joblib
import pickle
import regex as re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

with open(r"C:\Users\jeeva\Gen AI intern\eng_tokenizer.pkl", 'rb') as f:
    eng_tok = pickle.load(f)
with open(r"C:\Users\jeeva\Gen AI intern\fr_tokenizer.pkl", 'rb') as f:
    fr_tok = pickle.load(f)

gen_encoder = load_model(r"C:/Users/jeeva/Gen AI intern/encoder_model.h5")
gen_decoder = load_model(r"C:/Users/jeeva/Gen AI intern/decoder_model.h5")

app = Flask(__name__)

app.static_folder = 'static'


contractions = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have", "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
eng_stopwords = set(stopwords.words("english"))

eng_sequence_size = 10
sp_sequence_size = 20

def preprocess(sentence,language):
    sentence = sentence.lower()
    if language == "english":
        sentence = ' '.join([contractions[word] if word in contractions else word for word in sentence.split()])
#         sentence = ' '.join([word for word in sentence.split() if word not in eng_stopwords])
    sentence = re.sub(r"[.'!#$%&\'()*+,-./:;<=>?@[\\]^ `{|}~]"," ",sentence)
    sentence = ' '.join([word for word in sentence.split()])

    return sentence

def translate_user_input(user_input):
    preprocessed_input = preprocess(user_input, 'english')
    input_sequence = eng_tok.texts_to_sequences([preprocessed_input])
    padded_input_sequence = pad_sequences(input_sequence, padding='post', truncating='post', maxlen=eng_sequence_size)
    generated_translation = generate_from_encoder_input(padded_input_sequence)
    return generated_translation


def generate_from_encoder_input(encoder_input):
    encoder_input = encoder_input.reshape(1, -1)
    values, h, c = gen_encoder.predict(encoder_input)

    single_tok = np.zeros((1, 1))
    single_tok[0, 0] = fr_tok.word_index['sostoken']
    decoder_input = single_tok

    generated_sequence = []
    for count in range(sp_sequence_size):
        decoder_output, new_h, new_c = gen_decoder.predict([decoder_input] + [values, h, c])
        sampled_index = np.argmax(decoder_output[0, -1, :])
        sampled_word = fr_tok.index_word[sampled_index]

        if sampled_word != 'eostoken' and sampled_index != 0:
            generated_sequence.append(sampled_word)

        if sampled_word == 'eostoken':
            break

        h, c = new_h, new_c
        decoder_input[0, 0] = sampled_index

    generated_sequence = ' '.join(generated_sequence)
    return generated_sequence

@app.route('/')
def index():
    return render_template('index3.html')

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    user_message = data['user_message']

    translated_message = translate_user_input(user_message)
    response = {'sender': 'chatbot', 'message': translated_message}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)