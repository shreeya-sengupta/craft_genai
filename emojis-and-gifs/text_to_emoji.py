from transformers import pipeline
import emoji

from PIL import Image, ImageDraw, ImageFont
import imageio
import os

# Initialize a zero-shot classification pipeline for topic/keyword detection
classifier = pipeline("zero-shot-classification", model="<model_name>")

# emoji_map = {emoji.demojize(v.get('en')).strip(":") : v.get('en') for k, v in emoji.EMOJI_DATA.items()}
emoji_map = {}
emoji_map.update({
    "sad": ":disappointed:",
    "cat": ":cat:",
    "tree": ":deciduous_tree:",
    "love": ":heart:",
    "birthday": ":birthday:",
    "food": ":fork_and_knife:",
    "sleepy": ":sleeping:",
    "diwali": " Diwali :fireworks: :sparkler: :diya_lamp:"
})
candidate_labels = list(emoji_map.keys())
# print(emoji_map)


def text_to_emoji(user_text):
    # Classify text to find emoji-related keywords
    classified = classifier(user_text, candidate_labels)
    # Extract labels with high confidence
    labels = [label for label, score in zip(classified['labels'], classified['scores'])]
    # print(labels)

    # Replace keywords in text with emojis
    for label in labels:
    	if label in emoji_map:
    		user_text = user_text.replace(label, emoji.emojize(emoji_map[label]))

    return user_text

# GIF creator functions

def create_text_frame(text, font_size, width, height, bg_color, text_color):
    """Creates a single image frame with the given text."""
    img = Image.new('RGB', (width, height), color=bg_color)
    d = ImageDraw.Draw(img)

    # Calculate text position to center it
    bbox = d.textbbox((0, 0), text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) / 2
    y = (height - text_height) / 2 - bbox[1] # Adjust for font's baseline

    d.text((x, y), text, fill=text_color)
    return img

def create_gif_from_text(text_input, font_size, width, height, bg_color, text_color, duration_per_frame, output_filename="output.gif"):
    """Generates a GIF from the input text, character by character."""
    frames = []
    for i in range(1, len(text_input) + 1):
        current_text = text_input[:i]
        frame = create_text_frame(current_text, font_size, width, height, bg_color, text_color)
        frames.append(frame)

    imageio.mimsave(output_filename, frames, duration=duration_per_frame, loop=0) # loop=0 for infinite loop
    return output_filename
