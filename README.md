# COE379L-Final

# Objective
In a world dominated by using data to train machine learning models, how effective would it be to use synthetic data for such purposes? This project aims to quantify the resulting performance gap in this idea through a classification model approach. This problem is approached through a dual analysis of the CIFAR-10 data subset. Focusing strictly on the animal components, 3,000 synthetic images will be generated of each animal class. Using both the original and synthetic data, three experiments will be conducted and analyzed.
1. Train CNN on synthetic data, test on CIFAR-10 
2. Train CNN on CIFAR-10, test on synthetic data
3. Train and test CNN on a mixture of the two data types

# Contents
This project repository includes the following items.
1. Jupyter Notebooks containing model training, image generation, and experiments
2. Model inference server and deployment materials using Docker
3. Report summarizing methodology and results
4. Initial Project Proposal
5. AI usage document
6. Best performing models from the overall experiments 

# Jupyter Notebooks
`ExtractImages`: Notebook used to extract the real CIFAR-10 animal images.

`GenerateImage`: Notebook used to generate the 3,000 images of each animal class. The code in the notebook was reused for each animal class.

`Train_Synthetic`: Notebook containing the model training on the synthetic data, as well as the mixture of real and synthetic data.

`CIFAR-10_Train`: Notebook containing model training on the real CIFAR data and experiments related to the model. 

# Inference Server Setup
1. To begin the inference server for inferencing, the user must first pull the following Docker image `mjt2005/model_chaining:1.0` using the command `docker pull mjt2005/model_chaining:1.0`. You can also directly run the container with the command `docker run -it --rm -p 5000:5000 mjt2005/model_chaining:1.0` if you wish.
2. After pulling the premade image, the user must start the container for the server in the background using the command `docker compose up -d --build`.
3. Now the server is ready to take in user query routes.

# API Endpoints 
`curl -X POST http://localhost:5000/generate -H "Content-Type: application/json" -d '{"prompt": "blank"}' --output blank.jpg`: `POST` request that generates an image based on the prompt and saves to files as a jpg.

`curl localhost:5000/models/<model_type>`: `GET` request that returns the metadata of a selected model type: `mixed` or `real`. The 'mixed' model was trained on both synthetic and real images with a 55:45 split. The 'real' model was trained exclusively on real images.

`curl -X POST -F "image= blank.jpg" localhost:5000/inference/<model>`: `POST` request that returns the classification of the inputted image using the specified model in `<model>`.

# Stopping the Server
After the user has completed their analysis and experiment with our models, they should take down the container using the command `docker compose down`.

# Example Executions
Generating the Image:
`curl -X POST localhost:5000/generate "Content-Type: application/json" -d '{"prompt": "a squirrel in a spacesuit"}' --output space_squirrel.jpg`

Obtaining inference on the image with the 'mixed' model:
`curl -X POST -F "image=@img_2.jpg" localhost:5000/inference/mixed`

# References
1. The CIFAR-10 dataset can be accessed at this link: https://www.cs.toronto.edu/~kriz/cifar.html
2. The diffusion model used for generating the synthetic images can be found here: https://huggingface.co/stabilityai/sd-turbo


