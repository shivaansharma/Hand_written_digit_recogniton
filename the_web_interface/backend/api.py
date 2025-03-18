import matplotlib
matplotlib.use('Agg') 
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from Model import predict,initialize_parameters, load_weights, train, save_weights
import matplotlib.pyplot as plt

app = Flask(__name__)

# Allow CORS for frontend communication
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)



layers = [784, 256, 128, 10]

def preprocess_image(image_base64):
    """Preprocess the input image: Decode, resize, normalize."""
    image_data = base64.b64decode(image_base64.split(",")[1])
    image = Image.open(BytesIO(image_data)).convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image_array = np.array(image) / 255.0  # Normalize pixel values

    return image_array.reshape(1, -1), image_array  # Flatten + Original Image

def generate_visualization(image_array):
    """Generate a matplotlib plot and return it as a base64 string."""
    fig, ax = plt.subplots()
    ax.imshow(image_array)
    ax.set_title("Preprocessed Image")
    ax.axis('off')

    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight')
    plt.close(fig)

    return base64.b64encode(img_buffer.getvalue()).decode('utf-8')  # Convert to base64

@app.route('/api/train-model', methods=['POST'])
def train_model():
    W, b, l, ep = initialize_parameters(layer_sizes=layers)
    train(W, b, l, ep)
    save_weights(W,b)
    return jsonify({"message": "✅ Model trained and weights updated!"}), 200

@app.route('/api/process-image', methods=['POST'])
def process_image():
    data = request.json
    image_base64 = data.get('image')
    if not image_base64:
        return jsonify({"error": "No image provided"}), 400
    W,b = load_weights()
    image_array, original_image = preprocess_image(image_base64)
    predicted_label, probabilities = predict(image_array, W, b)
    
    # Generate Visualization
    visualization_base64 = generate_visualization(original_image)

    return jsonify({
        "message": "✅ Image processed",
        "prediction": int(predicted_label),
        "probabilities": probabilities.tolist()[0],  # Convert NumPy array to list
        "visualization": visualization_base64  # Base64-encoded visualization image
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)