import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences



model = load_model('language_transform.keras')
with open('src_tokenizer.json') as f:
    src_tokenizer = tokenizer_from_json(f.read())
with open('tar_tokenizer.json') as f:
    tar_tokenizer = tokenizer_from_json(f.read())

config = model.get_config()
max_no_of_words = int(config["layers"][0]["config"]["batch_shape"][1])


def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_seq(model, tokenizer, source):
    source = source.reshape((1, source.shape[0]))
    prediction = model.predict(source)
    integers = [np.argmax(vector) for vector in prediction[0]]
    target = []
    for i in integers:
        word = word_for_id(i, tokenizer)
        if word is None:
            break
        target.append(word)
    #Return Decoded Sentence
    return ' '.join(target)


tab1, tab2 = st.tabs(["English To French", "..."])
with tab1:
    st.header("Language translation using ML")
    input = st.text_input("Text to translate")
    if input:
        input = [input, "hi"]
        true_src = src_tokenizer.texts_to_sequences(input)
        source = pad_sequences(true_src, padding='post', maxlen=max_no_of_words)
        translation = predict_seq(model, tar_tokenizer, source[0])
        st.write(translation)


with tab2:
    st.text("Yet to come")
