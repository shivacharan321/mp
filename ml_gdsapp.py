import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
import os

# Load the pre-trained InceptionV3 model
model = InceptionV3(weights='imagenet')

def identify_medicinal_plant(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    img = image.img_to_array(img)
    img = preprocess_input(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

    predictions = model.predict(img)
    decoded_predictions = decode_predictions(predictions, top=1)[0]

    return decoded_predictions[0][1]

# Streamlit app
st.title("Medicinal Plant Identification")

# Upload image section
uploaded_file = st.file_uploader("Upload image of medicinal plant:", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Save the uploaded image
    image_path = 'uploaded_image.jpg'
    with open(image_path, 'wb') as f:
        f.write(uploaded_file.read())

    # Identify the plant from the uploaded image
    plant_name = identify_medicinal_plant(image_path)

    # Display the identified plant name
    st.success(f"The identified plant is: {plant_name}")
