import os
import logging
import warnings
import streamlit as st
import tensorflow as tf
from transformers import MarianTokenizer, TFMarianMTModel, GPT2Tokenizer, TFGPT2LMHeadModel

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Suppress TensorFlow and Transformers logs
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module='transformers')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='tensorflow')

# Set up logging to file
logging.basicConfig(filename='app.log', level=logging.DEBUG)

try:
    # Load pre-trained translation model and tokenizer
    logging.debug("Loading translation model and tokenizer")
    translation_model_name = 'Helsinki-NLP/opus-mt-en-fr'
    translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
    translation_model = TFMarianMTModel.from_pretrained(translation_model_name)
    logging.debug("Translation model and tokenizer loaded")

    # Load pre-trained style transfer model and tokenizer (assuming GPT-2 for simplicity)
    logging.debug("Loading style transfer model and tokenizer")
    style_model_name = 'gpt2-medium'
    style_tokenizer = GPT2Tokenizer.from_pretrained(style_model_name)
    style_model = TFGPT2LMHeadModel.from_pretrained(style_model_name)
    logging.debug("Style transfer model and tokenizer loaded")

    # Function to translate text
    def translate(text, model, tokenizer):
        inputs = tokenizer(text, return_tensors="tf", padding=True)
        translated_tokens = model.generate(**inputs)
        translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        return translated_text[0]

    # Function to apply style transfer
    def style_transfer(text, model, tokenizer, style_prompt):
        inputs = tokenizer.encode(style_prompt + text, return_tensors="tf")
        outputs = model.generate(inputs, max_length=200, num_return_sequences=1)
        styled_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return styled_text

    # Streamlit app
    st.title("Bilingual Stylistic Translator")

    # Input text
    text = st.text_input("Enter the text to be translated and styled:")

    if text:
        logging.debug("Translating text")
        # Translate text from English to French
        translated_text = translate(text, translation_model, translation_tokenizer)
        st.write("Translated Text:", translated_text)

        logging.debug("Applying style transfer")
        # Style transfer prompt
        style_prompt = "translate to formal: "

        # Apply style transfer (e.g., making the text more formal)
        styled_text = style_transfer(translated_text, style_model, style_tokenizer, style_prompt)
        st.write("Styled Text:", styled_text)

except Exception as e:
    logging.exception("Exception occurred")
    st.error("An error occurred. Please check the logs for more details.")
