from flask import Flask, Response, request, send_file
import numpy as np
import tensorflow as tf
import io
import logging
import uuid
from PIL import Image
import transformers
from transformers import pipeline
from diffusers import DiffusionPipeline
from diffusers import AutoPipelineForText2Image


app = Flask(__name__)

logging.basicConfig(level='DEBUG')

model10 = tf.keras.models.load_model('models/model10.keras')
real_model = tf.keras.models.load_model('models/lenet5_model_realCIFAR.keras')

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo", variant="fp16")
if pipe:
    logging.info("hf model successfully loaded")
      
def create_image(prompt):
    im = pipe(prompt=prompt, num_inference_steps=1, num_images_per_prompt=1, guidance_scale=0.0).images[0]
    return im

@app.route('/generate', methods=['POST'])
def generate_image():
    data = request.get_json()

    if not data or "prompt" not in data:
        return {"error": "Missing JSON field `prompt`"}, 400

    prompt = data["prompt"]

    try:
        # Generate image
        im = create_image(prompt)

        # Create a unique filename to avoid collisions
        file_id = str(uuid.uuid4())
        filename = f"generated_{file_id}.jpg"

        # Save image to disk
        im.save(f"/images/{filename}", format="JPEG")

        # Return the file just like your job code
        return send_file(f"/images/{filename}", mimetype="image/jpeg", as_attachment=True)

    except Exception as e:
        return {"error": f"Image generation failed: {e}"}, 500


@app.route('/models/<model_type>', methods=['GET'])
def model_info(model_type):
    if model_type == "real":
        return {
      "model" : "Lenet_5 arhcitecture trained on 100% real data",
      "version": "v1",
      "name": "animal_classifier",
      "description": "Classify dog, cat, bird, frog, horse, and deer images.",
      "number_of_parameters": 386694
   }
    elif model_type == "mixed":
        return {
      "model" : "Lenet_5 arhcitecture trained on 46% real data",
      "version": "v1",
      "name": "animal_classifier",
      "description": "Classify dog, cat, bird, frog, horse, and deer images.",
      "number_of_parameters": 386694
   }
        
def preprocess_image(im):
    im = im.resize((32,32), Image.LANCZOS)
    arr = np.array(im)
    arr = arr/255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.route('/inference/<model>', methods=['POST'])
def classify(model):
   # check if the post request has the file part
   if 'image' not in request.files:
      return {"error": "Invalid request; pass a binary image file as a multi-part form under the image key."}
   # get the data
   im = request.files['image']
   
   if not im:
      return {"error": "The `image` field is required"}, 404

   try:   
       image_stream = io.BytesIO(im.read())
       im = Image.open(image_stream)
   except Exception as e:
       return {"error": f"Could not read file bytes details: {e}"}, 404
   try:
      data = preprocess_image(im)
   except Exception as e:
      return {"error": f"Could not process the `image` field; details: {e}"}, 404

   if model == "mixed":
       result = model10.predict(data)
   elif model == "real":
       result = real_model.predict(data)
  
   prediction = np.argmax(result, axis = 1)
   if prediction == 0:
       classification = 'bird'
   elif prediction == 1:
       classification = 'cat'
   elif prediction == 2:
       classification = 'deer'
   elif prediction == 3:
       classification = 'dog'
   elif prediction == 4:
       classification = 'frog'
   elif prediction == 5:
       classification = 'horse'
   else:
       return {"error" : "No classification"}
       
   return {"prediction": classification}
    
# start the development server
if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0')