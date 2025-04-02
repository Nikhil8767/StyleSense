import streamlit as st
import os
import pathlib
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

BASE_DIR = pathlib.Path(__file__).parent.resolve()
feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')

# def save_uploaded_file(uploaded_file):
#     try:
#         with open(os.path.join('uploads',uploaded_file.name),'wb') as f:
#             f.write(uploaded_file.getbuffer())
#         return 1
#     except:
#         return 0

def save_uploaded_file(uploaded_file):
    try:
        os.makedirs('uploads', exist_ok=True)  # Ensure folder exists
        file_path = os.path.join('uploads', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path  # Return saved file path
    except Exception as e:
        st.error(f"File upload failed: {e}")
        return None



# def feature_extraction(img_path,model):
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     expanded_img_array = np.expand_dims(img_array, axis=0)
#     preprocessed_img = preprocess_input(expanded_img_array)
#     result = model.predict(preprocessed_img).flatten()
#     normalized_result = result / norm(result)
#
#     return normalized_result


def feature_extraction(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    return result / norm(result)

# def recommend(features,feature_list):
#     neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
#     neighbors.fit(feature_list)
#
#     distances, indices = neighbors.kneighbors([features])
#
#     return indices


def recommend(features, feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

# steps
# file upload -> save
# uploaded_file = st.file_uploader("Choose an image")
# if uploaded_file is not None:
#     if save_uploaded_file(uploaded_file):
#         # display the file
#         display_image = Image.open(uploaded_file)
#         st.image(display_image)
#         # feature extract
#         features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
#         #st.text(features)
#         # recommendention
#         indices = recommend(features,feature_list)
#         # show
#         col1,col2,col3,col4,col5 = st.beta_columns(5)
#
#         with col1:
#             st.image(filenames[indices[0][0]])
#         with col2:
#             st.image(filenames[indices[0][1]])
#         with col3:
#             st.image(filenames[indices[0][2]])
#         with col4:
#             st.image(filenames[indices[0][3]])
#         with col5:
#             st.image(filenames[indices[0][4]])
#     else:
#         st.header("Some error occured in file upload")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    if file_path:
        # Display uploaded image
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption="Uploaded Image", use_container_width=True)

        # Feature extraction
        features = feature_extraction(file_path, model)

        # Recommendations
        indices = recommend(features, feature_list)

        # Display recommended images
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(filenames[indices[0][0]], caption="Similar 1")
        with col2:
            st.image(filenames[indices[0][1]], caption="Similar 2")
        with col3:
            st.image(filenames[indices[0][2]], caption="Similar 3")
        with col4:
            st.image(filenames[indices[0][3]], caption="Similar 4")
        with col5:
            st.image(filenames[indices[0][4]], caption="Similar 5")
    else:
        st.error("File upload failed. Please try again.")
