
import numpy as np 
import pandas as pd 


import tensorflow as tf
from tensorflow import keras

import re
from nltk.corpus import stopwords

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")




df = pd.read_csv("D:\data\eng_-french.csv",names=['english','french'])
print(df.head(5))

print(f"\n\n Shape of the data >>{df.shape}")
df.sample(5)


print("Unique values before dropping duplicates")
print(df.english.nunique())
print(df.french.nunique())

df.drop_duplicates(subset=['english'],inplace=True)
df.drop_duplicates(subset=['french'],inplace=True)

print("\n\nUnique values before dropping duplicates")
print(df.english.nunique())
print(df.french.nunique())


print("Checking NA values\n")
print(df.isnull().any(),'\n')
print(df.isnull().sum())


print("before preprocessing")
df.tail(6)



import nltk

nltk.download('stopwords')


contractions = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have", "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}
eng_stopwords = set(stopwords.words("english"))

def preprocess(sentence,language):
    sentence = sentence.lower()
    if language == "english":
        sentence = ' '.join([contractions[word] if word in contractions else word for word in sentence.split()])
#         sentence = ' '.join([word for word in sentence.split() if word not in eng_stopwords])
    sentence = re.sub(r"[.'!#$%&\'()*+,-./:;<=>?@[\\]^ `{|}~]"," ",sentence)
    sentence = ' '.join([word for word in sentence.split()])

    return sentence



df.english = df.english.apply(lambda x:preprocess(x,'english'))
df.french = df.french.apply(lambda x:preprocess(x,'french'))

print(df.shape,'\n')
df.info()



print("after preprocessing")
df.tail(6)



df["french_input"] = df.french.apply(lambda x:'sostoken ' + x)
df["french_label"] = df.french.apply(lambda x:x + ' eostoken')

encoder_input = np.array(df.english)
decoder_input = np.array(df.french_input)
decoder_label = np.array(df.french_label)


indices = np.arange(116544)
np.random.shuffle(indices)

encoder_input = encoder_input[indices]
decoder_input = decoder_input[indices]
decoder_label = decoder_label[indices]

df.head()



total = df.shape[0]
test_size = 0.3

train_encoder_input = encoder_input[:-int(total*test_size)]
train_decoder_input = decoder_input[:-int(total*test_size)]
train_decoder_label = decoder_label[:-int(total*test_size)]

test_encoder_input = encoder_input[-int(total*test_size):]
test_decoder_input = decoder_input[-int(total*test_size):]
test_decoder_label = decoder_label[-int(total*test_size):]

print("train dataset shape")
print(train_encoder_input.shape)
print(train_decoder_input.shape)
print(train_decoder_label.shape)

print("\n\ntest dataset shape")
print(test_encoder_input.shape)
print(test_decoder_input.shape)
print(test_decoder_label.shape)



eng_tok = Tokenizer()
eng_tok.fit_on_texts(train_encoder_input)
print(f"Number of unique words used in english sentences >> {len(eng_tok.index_word)}")

fr_tok = Tokenizer()
fr_tok.fit_on_texts(train_decoder_input)
fr_tok.fit_on_texts(train_decoder_label)
print(f"Number of unique words used in french sentences >> {len(fr_tok.index_word)}")



total_counts = 0
rare_counts = 0
total_freq = 0
rare_freq = 0

least_occurence = 3
for k,v in eng_tok.word_counts.items():
    total_counts +=1
    total_freq += v
    if v < least_occurence:
        rare_counts+=1
        rare_freq += v

print("="*25,"english","="*25)
print(f"{rare_counts} of {total_counts} words are used less than {least_occurence}times,")
print(f"which is only {np.round(rare_counts/total_counts*100)}% of total words used")
print(f"But they occupy {np.round(rare_freq/total_freq*100)}% of total frequency ")



total_counts = 0
rare_counts = 0
total_freq = 0
rare_freq = 0

least_occurence = 3
for k,v in fr_tok.word_counts.items():
    total_counts +=1
    total_freq += v
    if v < least_occurence:
        rare_counts+=1
        rare_freq += v

print("="*25,"french","="*25)
print(f"{rare_counts} of {total_counts} words are used less than {least_occurence}times,")
print(f"which is only {np.round(rare_counts/total_counts*100)}% of total words used")
print(f"But they occupy {np.round(rare_freq/total_freq*100)}% of total frequency ")



eng_word_size = 6000
eng_vocab_size = eng_word_size+1
fr_word_size = 12000
fr_vocab_size = fr_word_size+1

eng_tok = Tokenizer(num_words=eng_word_size)
eng_tok.fit_on_texts(train_encoder_input)

train_encoder_input = eng_tok.texts_to_sequences(train_encoder_input)
test_encoder_input = eng_tok.texts_to_sequences(test_encoder_input)

fr_tok = Tokenizer(num_words=fr_word_size)
fr_tok.fit_on_texts(train_decoder_input)
fr_tok.fit_on_texts(train_decoder_label)

train_decoder_input = fr_tok.texts_to_sequences(train_decoder_input)
train_decoder_label = fr_tok.texts_to_sequences(train_decoder_label)

test_decoder_input = fr_tok.texts_to_sequences(test_decoder_input)
test_decoder_label = fr_tok.texts_to_sequences(test_decoder_label)


'''

print("english")
eng_lens = [len(seq) for seq in train_encoder_input]
print("mean >> ",np.mean(eng_lens))
plt.subplot(2,1,1)
plt.hist(eng_lens,bins=50)


print("french")
fr_lens = [len(seq) for seq in train_decoder_input]
print("mean >> ",np.mean(fr_lens))
plt.subplot(2,1,2)
plt.hist(fr_lens,bins=50)
plt.show()'''

eng_sequence_size = 10
fr_sequence_size = 20

train_encoder_input = pad_sequences(train_encoder_input,padding='post',truncating='post',maxlen=eng_sequence_size)
test_encoder_input = pad_sequences(test_encoder_input,padding='post',truncating='post',maxlen=eng_sequence_size)

train_decoder_input = pad_sequences(train_decoder_input,padding='post',truncating='post',maxlen=fr_sequence_size)
train_decoder_label = pad_sequences(train_decoder_label,padding='post',truncating='post',maxlen=fr_sequence_size)

test_decoder_input = pad_sequences(test_decoder_input,padding='post',truncating='post',maxlen=fr_sequence_size)
test_decoder_label = pad_sequences(test_decoder_label,padding='post',truncating='post',maxlen=fr_sequence_size)

print("train dataset shape")
print(train_encoder_input.shape)
print(train_decoder_input.shape)
print(train_decoder_label.shape)

print("\n\ntest dataset shape")
print(test_encoder_input.shape)
print(test_decoder_input.shape)
print(test_decoder_label.shape)




from keras.layers import Input,Embedding,LSTM,Dense,Concatenate,Attention
from keras.models import Model
from keras.utils import plot_model
from keras import backend as K

#hyperparameters
embedding_size = 256
hidden_size = 256

encoder_input = Input(shape=[eng_sequence_size])
encoder_embedding = Embedding(eng_vocab_size,embedding_size,mask_zero=True)
encoder_embedded = encoder_embedding(encoder_input)

encoder_lstm1 = LSTM(hidden_size,return_sequences=True,return_state=True,dropout=0.2,recurrent_dropout=0.2)
encoder_output1,encoder_h1,encoder_c1 = encoder_lstm1(encoder_embedded)

encoder_lstm2 = LSTM(hidden_size,return_sequences=True,return_state=True,dropout=0.2,recurrent_dropout=0.2)
encoder_output2,encoder_h2,encoder_c2 = encoder_lstm2(encoder_output1)

encoder_lstm3 = LSTM(hidden_size,return_sequences=True,return_state=True,dropout=0.2,recurrent_dropout=0.2)
encoder_output3,encoder_h3,encoder_c3 = encoder_lstm3(encoder_output1)

decoder_input = Input(shape=(None,))
decoder_embedding = Embedding(fr_vocab_size,embedding_size,mask_zero=True)
decoder_embedded = decoder_embedding(decoder_input)

decoder_lstm = LSTM(hidden_size,return_sequences=True,return_state=True,dropout=0.2,recurrent_dropout=0.2)
decoder_output,_,_ = decoder_lstm(decoder_embedded,initial_state=[encoder_h3,encoder_c3])

attn_layer = Attention()
attn_context = attn_layer([decoder_output,encoder_output3])

decoder_output = Concatenate(axis=-1)([decoder_output,attn_context])
tanh_dense= Dense(hidden_size*2,activation=K.tanh)
decoder_output = tanh_dense(decoder_output)

softmax_dense = Dense(fr_vocab_size,activation='softmax')
decoder_output = softmax_dense(decoder_output)

trainer_model = Model([encoder_input,decoder_input],decoder_output)
trainer_model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])




trainer_hist =trainer_model.fit([train_encoder_input,train_decoder_input],train_decoder_label,epochs=10,batch_size=128,validation_split=0.2)

gen_encoder = Model([encoder_input],[encoder_output3,encoder_h3,encoder_c3])

gen_decoder_values_input = Input(shape=(eng_sequence_size,hidden_size))
gen_decoder_h_input = Input(shape=[hidden_size])
gen_decoder_c_input = Input(shape=[hidden_size])

gen_decoder_embedded = decoder_embedding(decoder_input)
gen_decoder_output,gen_decoder_h,gen_decoder_c = decoder_lstm(gen_decoder_embedded,initial_state=[gen_decoder_h_input,gen_decoder_c_input])

attn_context = attn_layer([gen_decoder_output,gen_decoder_values_input])
gen_decoder_output = Concatenate(axis=-1)([gen_decoder_output,attn_context])

gen_decoder_output = tanh_dense(gen_decoder_output)
gen_decoder_output = softmax_dense(gen_decoder_output)

gen_decoder = Model([decoder_input]+[gen_decoder_values_input,gen_decoder_h_input,gen_decoder_c_input],[gen_decoder_output,gen_decoder_h,gen_decoder_c])




def seq2eng(seq):
    ret =[]
    for n in seq:
        if n != 0:
            ret.append(eng_tok.index_word[n])
    ret = ' '.join(ret)
    return ret

def seq2fr(seq):
    ret =[]
    for n in seq:
        if n != 0 and fr_tok.index_word[n] != 'eostoken':
            ret.append(fr_tok.index_word[n])
    ret = ' '.join(ret)
    return ret



def generate_from_encoder_input(encoder_input):
    encoder_input = encoder_input.reshape(1,-1)
    values,h,c = gen_encoder.predict(encoder_input)

    single_tok = np.zeros((1,1))
    single_tok[0,0] = fr_tok.word_index['sostoken']
    decoder_input = single_tok

    generated = []
    count = 0
    while(True):
        decoder_output,new_h,new_c = gen_decoder.predict([decoder_input]+[values,h,c])
        count +=1

        sampled_index = np.argmax(decoder_output[0,-1,:])
        sampled_word = fr_tok.index_word[sampled_index]

        if sampled_word != 'eostoken' and sampled_index != 0:
            generated.append(sampled_word)
        if count >= fr_sequence_size or sampled_word == 'eostoken':
            break

        h,c = new_h,new_c
        decoder_input[0,0] = sampled_index

    generated = ' '.join(generated)
    return generated




for i in range(520,525):
    print("\n<<sample encoder input english sentence>>")
    print(seq2eng(train_encoder_input[i]))
    print("\n")
    print("<<sample generated french sentence>>")
    print(generate_from_encoder_input(train_encoder_input[i]))
    print("\n")
    print("<<answer french sentence>>")
    print(seq2fr(train_decoder_label[i]))
    print("========================================\n")




idx = [24,1525,666,2222,52212]

print("Results on Train Dataset")
for i in idx:
    print("\nINPUT ENG>>")
    print(seq2eng(train_encoder_input[i]))
    print("\n")
    print("GENERATED FR>>")
    print(generate_from_encoder_input(train_encoder_input[i]))
    print("\n")
    print("ANSWER FR>>")
    print(seq2fr(train_decoder_label[i]))
    print("=====================================================================\n")



idx = [24,1525,666,2222]

print("Results on Test Dataset")
for i in idx:
    print("\nINPUT ENG>")
    print(seq2eng(test_encoder_input[i]))
    print(test_encoder_input[i])
    print("\n")
    print("<<GENERATED FR>>")
    print(generate_from_encoder_input(test_encoder_input[i]))
    print("\n")
    print("<<ANSWER FR>>")
    print(seq2fr(test_decoder_label[i]))
    print(test_decoder_label[i])
    print("=====================================================================\n")



def translate_user_input(user_input):
    # Preprocess user input
    preprocessed_input = preprocess(user_input, 'english')

    # Tokenize and pad user input
    input_sequence = eng_tok.texts_to_sequences([preprocessed_input])
    padded_input_sequence = pad_sequences(input_sequence, padding='post', truncating='post', maxlen=eng_sequence_size)

    # Generate translation
    generated_translation = generate_from_encoder_input(padded_input_sequence)

    return generated_translation




user_input_sentence = input("enter : ")
translated_sentence = translate_user_input(user_input_sentence)

print(f"English: {user_input_sentence}")
print(f"French Translation: {translated_sentence}")



from keras.models import load_model



gen_encoder.save('encoder_model.h5')


gen_decoder.save('decoder_model.h5')



import pickle
with open('eng_tokenizer.pkl', 'wb') as f:
    pickle.dump(eng_tok, f)



with open('fr_tokenizer.pkl', 'wb') as f:
    pickle.dump(fr_tok, f)




