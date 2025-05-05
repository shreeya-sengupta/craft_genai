# import the streamlit library
import streamlit as st
from bpe import apply_bpe, final_vocab, learned_merges

# give a title to our app
st.title('Tokenizer')

# TAKE WEIGHT INPUT in kgs
text_to_tokenize = st.text_input("Enter text to tokenize")

# TAKE HEIGHT INPUT
# radio button to choose height format
status = st.radio('Select Tokenizer: ',
                  ('BPE'))

# compare status value
if(status == 'BPE'):
    # take height input in centimeters
    try:
    	words = text_to_tokenize.split()
    	tokens = [apply_bpe(word, learned_merges) for word in words]
    	# for word, tkns in zip(words, tokens):
    	# 	print(f"{word:8} => {' '.join(tkns)}")
    except:
        st.text("select a tokenizer")

# check if the button is pressed or not
if(st.button('Tokenize')):

    # print the BMI INDEX
    st.text("tokens {}.".format(tokens))
