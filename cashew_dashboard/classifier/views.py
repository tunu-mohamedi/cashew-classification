import os
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
MODEL_PATH = os.path.join(os.path.dirname(settings.BASE_DIR), 'cashew_nut_disease_model.keras')
class_names = ['anthracnose', 'gumosis', 'healthy', 'leaf miner', 'red rust']

# Attempt to load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Error loading model from {MODEL_PATH}: {e}")

def index(request):
    context = {}
    if request.method == 'POST' and request.FILES.get('upload'):
        if model is None:
            context['error'] = "Model not loaded. Please train and save the model first."
            return render(request, 'classifier/index.html', context)

        uploaded_file = request.FILES['upload']
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        url = fs.url(name)
        file_path = fs.path(name)

        try:
            # Process image matching training parameters (224x224)
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            # Predictions
            predictions = model.predict(img_array)
            # The training used SparseCategoricalCrossentropy(from_logits=True)
            score = tf.nn.softmax(predictions[0])
            predicted_index = np.argmax(score)
            predicted_class = class_names[predicted_index]
            confidence = round(float(100 * np.max(score)), 2)

            threshold = 70.0  # Confidence threshold (percent)
            context['url'] = url
            context['confidence'] = confidence
            if confidence < threshold:
                context['predicted_class'] = 'Not a cashew or cashew leaf'
                context['flagged'] = True
            else:
                context['predicted_class'] = predicted_class
                context['flagged'] = False
        except Exception as e:
            context['error'] = f"Prediction failed: {str(e)}"

    return render(request, 'classifier/index.html', context)
