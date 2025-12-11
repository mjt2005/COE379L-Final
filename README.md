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

# Jupyter Notebooks
`ExtractImages`: Notebook used to extract the real CIFAR-10 animal images.

`GenerateImage`: Notebook used to generate the 3,000 images of each animal class. The code in the notebook was reused for each animal class.

`Train_Synthetic`: Notebook containing the model training on the synthetic data, as well as the mixture of real and synthetic data.

`CIFAR-10_Train`: Notebook containing model training on the real CIFAR data and experiments related to the model. 



