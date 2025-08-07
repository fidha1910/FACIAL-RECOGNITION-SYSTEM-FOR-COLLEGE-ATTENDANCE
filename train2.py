import os
import face_recognition

def train_face_recognition_model(faces_folder):
    """
    Trains a face recognition model from images in a folder.

    Args:
        faces_folder (str): The path to the folder containing subfolders for each person's images.

    Returns:
        tuple: A tuple containing two lists: known_face_encodings and known_face_names.
    """
    known_face_encodings = []
    known_face_names = []
    faces_folder = "needed set"

    if not os.path.exists(faces_folder):
        os.makedirs(faces_folder)

    print("Starting training...")
    for person_name in os.listdir(faces_folder):
        person_folder = os.path.join(faces_folder, person_name)
        if os.path.isdir(person_folder):
            print(f"Training Started for {person_name}")
            for filename in os.listdir(person_folder):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(person_folder, filename)
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        known_face_encodings.append(encodings[0])
                        known_face_names.append(person_name)
                    else:
                        print(f"Warning: No faces found in {filename}")

    print("Training completed.")
    return known_face_encodings, known_face_names

# Example usage (assuming you have a 'faces' folder with subfolders):
faces_folder = "faces" # change "faces" to the path of your folder
known_face_encodings, known_face_names = train_face_recognition_model(faces_folder)

# Now you have your trained data in known_face_encodings and known_face_names!
# You can save this data to a file if you want to use it later.
# For example, using pickle:
import pickle

model_data = {"encodings": known_face_encodings, "names": known_face_names}

with open("face_recognition_model.pkl", "wb") as f:
  pickle.dump(model_data, f)
print("Model Saved to face_recognition_model.pkl")

# To load the model later:
# with open("face_recognition_model.pkl", "rb") as f:
#     loaded_model_data = pickle.load(f)
#     loaded_encodings = loaded_model_data["encodings"]
#     loaded_names = loaded_model_data["names"]