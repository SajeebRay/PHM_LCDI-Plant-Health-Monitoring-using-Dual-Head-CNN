import streamlit as st
from PIL import Image
from model_for_app import load_model_and_labels, predict_image
from utils import read_imagefile
import torch
import os
import io

# Load model and labels
model, Names, diseases = load_model_and_labels()

# Mapping plant to valid diseases
plant_disease_map = {
    "Apple": [0, 1, 2, 4],
    "Blueberry": [3],
    "Cherry": [5, 6],
    "Corn": [7, 8, 9, 10],
    "Grape": [11, 12, 13, 14],
    "Orange": [15],
    "Peach": [16, 17],
    "Pepper": [18, 19],
    "Potato": [20, 21, 22],
    "Raspberry": [23],
    "Soybean": [24],
    "Squash": [25],
    "Strawberry": [26, 27],
    "Tomato": [28, 29, 30, 31, 32, 33, 34, 35, 36, 37],
}

# Streamlit app config
# st.set_page_config(page_title="Plant Health Monitoring", layout="centered")
# st.title("🌿 Plant Health Monitoring")
# st.markdown(
#     "Upload an image of a plant leaf to detect the **plant type** and **possible disease**."
# )

# Streamlit app config
st.set_page_config(page_title="Plant Health Monitoring", layout="centered")

# Title and Description
st.title("🌿 Plant Health Monitoring")
st.markdown("""
Upload an image of a **plant leaf** to automatically detect the **plant type** and identify any **diseases** present.  
This tool uses a trained AI model to assist farmers, gardeners, and researchers in monitoring plant health and making informed decisions.
""")

# Divider
st.markdown("---")

# Supported Plant Names and Diseases
st.header("🧬 Supported Plant Types and Diseases")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🌱 Plant Types")
    plant_names = [
        "Apple", "Blueberry", "Cherry", "Corn", "Grape", "Orange", "Peach",
        "Pepper", "Potato", "Raspberry", "Soybean", "Squash", "Strawberry", "Tomato"
    ]
    st.markdown("\n".join([f"- {name}" for name in plant_names]))

with col2:
    st.subheader("🦠 Detectable Leaf Types")
    diseases = [
        "Apple Black Rot", "Apple Healthy", "Apple Scab", "Blueberry Healthy", "Cedar Apple Rust",
        "Cherry Healthy", "Cherry Powdery Mildew", "Corn Cercospora Leaf Spot", "Corn Common Rust",
        "Corn Healthy", "Corn Northern Leaf Blight", "Grape Black Rot", "Grape Esca",
        "Grape Healthy", "Grape Leaf Blight", "Orange Haunglongbing", "Peach Bacterial Spot",
        "Peach Healthy", "Pepper Bacterial Spot", "Pepper Healthy", "Potato Early Blight",
        "Potato Healthy", "Potato Late Blight", "Raspberry Healthy", "Soybean Healthy",
        "Squash Powdery Mildew", "Strawberry Healthy", "Strawberry Leaf Scorch",
        "Tomato Bacterial Spot", "Tomato Early Blight", "Tomato Late Blight", "Tomato Leaf Mold",
        "Tomato Mosaic Virus", "Tomato Septoria Leaf Spot", "Tomato Spider Mites",
        "Tomato Target Spot", "Tomato Yellow Leaf Curl Virus", "Tomato Healthy"
    ]
    st.markdown("\n".join([f"- {disease}" for disease in diseases]))

st.warning("⚠️ Note: Uploading a image from an **unsupported or unlisted plant type** may result in **inaccurate or incorrect predictions**.")

#########Example ################
use_examples = st.checkbox("Use Examples")

uploaded_file = None

if use_examples:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    example_dir = os.path.join(script_dir, "examples")
    if os.path.isdir(example_dir):
        example_files = [f for f in os.listdir(example_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if example_files:
            st.subheader("📷 Click an Example Image to Predict")
            cols = st.columns(min(4, len(example_files)))  # display up to 4 per row

            for idx, file in enumerate(example_files):
                img_path = os.path.join(example_dir, file)
                try:
                    img = Image.open(img_path)
                except Exception as e:
                    st.warning(f"Cannot open {file}: {e}")
                    continue

                # Display image
                cols[idx % 4].image(img, use_column_width=True)

                # Make the image clickable using a button with unique key
                if cols[idx % 4].button(" ", key=f"{file}_btn"):
                    with open(img_path, "rb") as f:
                        uploaded_file = io.BytesIO(f.read())

else:
    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

# Prediction code (same as before)
if uploaded_file is not None:
    try:
        image_bytes = uploaded_file.read()
        image = read_imagefile(image_bytes)
        st.image(Image.open(io.BytesIO(image_bytes)), caption="Selected Image", use_column_width=True)

        leaf_class, disease_class, leaf_conf, disease_conf, disease_probs = predict_image(image, model, return_probs=True)

        plant_name = Names[leaf_class]
        valid_indices = plant_disease_map.get(plant_name, [])

        if valid_indices:
            masked_probs = torch.tensor(
                [disease_probs[i] if i in valid_indices else float("-inf") for i in range(len(disease_probs))]
            )
            disease_class = int(torch.argmax(masked_probs).item())
            disease_conf = disease_probs[disease_class]

        st.success("✅ Prediction Complete")
        st.write("### Results:")
        st.write(f"**Plant Name**: {plant_name} ({leaf_conf * 100:.2f}% confidence)")
        st.write(f"**Disease**: {diseases[disease_class]} ({disease_conf * 100:.2f}% confidence)")

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")

################Example End #########
# uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     try:
#         image_bytes = uploaded_file.read()
#         image = read_imagefile(image_bytes)
#         st.image(
#             Image.open(uploaded_file), caption="Uploaded Image", use_column_width=True
#         )

#         leaf_class, disease_class, leaf_conf, disease_conf, disease_probs = (
#             predict_image(image, model, return_probs=True)
#         )

#         plant_name = Names[leaf_class]
#         valid_indices = plant_disease_map.get(plant_name, [])

#         if valid_indices:
#             masked_probs = torch.tensor(
#                 [
#                     disease_probs[i] if i in valid_indices else float("-inf")
#                     for i in range(len(disease_probs))
#                 ]
#             )
#             disease_class = int(torch.argmax(masked_probs).item())
#             disease_conf = disease_probs[disease_class]

#         st.success("✅ Prediction Complete")
#         st.write("### Results:")
#         st.write(f"**Plant Name**: {plant_name} ({leaf_conf * 100:.2f}% confidence)")
#         st.write(
#             f"**Disease**: {diseases[disease_class]} ({disease_conf * 100:.2f}% confidence)"
#         )

#     except Exception as e:
#         st.error(f"Error during prediction: {str(e)}")

st.markdown("""
---  
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    Developed by <strong>Sajeeb Kumar Ray</strong> <br>
    <em>Deep Vision Research Lab (DVR Lab)</em>
    <br>
    <em>Department of Information and Communication Engineering (ICE) </em>
    <br>
    <em>Pabna University of Science and Technology, Pabna, Bangladesh</em>
</div>
""", unsafe_allow_html=True)


# if __name__ == "__main__":
#     import os
#     os.system("streamlit run " + __file__)
