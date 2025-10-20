# Streamlit UI

import streamlit as st
from text_to_emoji import text_to_emoji
from text_to_emoji import create_gif_from_text

# RESULT = str()

st.title("Emojis & GIFs")
user_input = st.text_area("Enter your text here:", height=100)

if st.button("Generate Emojis"):
    if user_input.strip():
    	with st.spinner("Processing data... Please wait."):
        	st.session_state.RESULT = text_to_emoji(user_input.lower())
    	st.success(st.session_state.RESULT)
    else:
        st.warning("Please enter some text to convert.")

# create gif from text 
# GIF customization options
col1, col2 = st.columns(2)
with col1:
    font_size = st.slider("Font Size", 20, 100, 50)
    width = st.slider("GIF Width", 200, 800, 400)
    bg_color = st.color_picker("Background Color", "#FFFFFF")
with col2:
    duration_per_frame = st.slider("Duration per character (ms)", 50, 500, 100) / 1000.0 # Convert to seconds
    height = st.slider("GIF Height", 100, 400, 200)
    text_color = st.color_picker("Text Color", "#000000")


if st.button("Generate GIF"):
	if st.session_state.RESULT:
	    with st.spinner("Generating GIF..."):
	        gif_filename = create_gif_from_text(st.session_state.RESULT, font_size, width, height, bg_color, text_color, duration_per_frame)
	        st.success("GIF generated successfully!")
	        st.image(gif_filename, caption="Generated GIF", use_container_width=True)

	        with open(gif_filename, "rb") as file:
	            btn = st.download_button(
	                label="Download GIF",
	                data=file,
	                file_name="generated_text.gif",
	                mime="image/gif"
	            )
	else:
	    st.warning("Please enter some text to generate a GIF.")