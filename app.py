from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from semantic_search import text_to_image, image_to_image, hybrid_query, pca_image_to_image
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

# Define the main route
@app.route('/')
def index():
    return render_template('index.html')


@app.route("/text_to_img", methods=["POST"])
def text_to_img():
    if request.method == "POST":
        text = request.form["text_input"]
        results = text_to_image(text)
        return render_template("index.html", results=results)

# Allowed file extensions for images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/image_to_img", methods=["POST", "GET"])
def image_to_img():
    if request.method == "POST":
        # Check if the 'image_input' field is in the form data
        if 'image_input' not in request.files:
            return "No file part", 400
        
        file = request.files['image_input']
        
        # If no file is selected
        if file.filename == '':
            return "No selected file", 400
        
        # If the file is allowed, save it and process
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_path = os.path.join('uploads', filename)
            file.save(image_path)  # Save the uploaded image to the 'uploads' folder

            # Call your image_to_image function to process the image
            results = image_to_image(image_path)

            return render_template("index.html", results=results)
        else:
            return "Invalid file type", 400
    return render_template("index.html", results=None)

@app.route("/hybrid_query", methods=["POST", "GET"])
def hybrid_query_route():
    if request.method == "POST":
        # Check if the 'image_input' field is in the form data
        if 'image_input' not in request.files or 'text_input' not in request.form:
            return "No file or text input", 400

        # Get the image file
        file = request.files['image_input']
        if file.filename == '':
            return "No selected file", 400

        # Ensure the file is allowed
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_path = os.path.join('uploads', filename)
            file.save(image_path)  # Save the image to the 'uploads' folder

            # Get the text input from the form
            text_input = request.form['text_input']

            # Call the hybrid_query function with the uploaded image and text input
            results = hybrid_query(image_path, text_input)

            return render_template("index.html", results=results)

        else:
            return "Invalid file type", 400

    return render_template("index.html", results=None)

@app.route("/pca_image_to_image", methods=["POST", "GET"])
def pca_image_to_image_route():
    if request.method == "POST":
        # Check if the 'image_input' field is in the form data
        if 'image_input' not in request.files or 'text_input' not in request.form:
            return "No file or text input", 400

        # Get the image file
        file = request.files['image_input']
        if file.filename == '':
            return "No selected file", 400

        # Ensure the file is allowed
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            image_path = os.path.join('uploads', filename)
            file.save(image_path)  # Save the image to the 'uploads' folder

            # Get the text input from the form
            k = request.form['text_input']

            # Call the hybrid_query function with the uploaded image and text input
            results = pca_image_to_image(image_path, int(k))

            return render_template("index.html", results=results)

        else:
            return "Invalid file type", 400

    return render_template("index.html", results=None)

if __name__ == "__main__":
    app.run(port = 5000, debug=True)