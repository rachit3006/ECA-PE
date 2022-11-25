# Emotion Recognition Website

A flask-based website deployed with an emotion recognition model built in python referring to the research paper present in the repository. The website captures frames
from user's webcam and runs the model on the captured frames to output the detected emotion and engagement status (based on the emotion detected) of the user.

## Model Description
<p>The model is built using VGGFace2-model. The output of the VGGFace2-model is passed to three more layers as follows -
<ol>
<li> A 1024 neurons dense layer with relu activation 
<li> A dropout layer
<li> A 7 neurons dense layer with softmax activation
</ol>
The last layer with the softmax activation function allows the model to predict the probability of 7 emotional 
categories namely angry, disgust, fear, happy, neutral, sad, and surprise.</p>

## Datasets Used
<p>The datasets used to train and test the model (the combined data was split to form train and test data) are -
<ol>
<li> Affectnet - https://www.kaggle.com/datasets/tom99763/affectnethq
<li> FER2013 - https://www.kaggle.com/datasets/msambare/fer2013 
</ol>
The AffectNet class distribution is strongly shifted to the neutral and happiness classes, which introduces a class imbalance issue. Thus, to solve this issue 
the data from the FER2013 dataset containing the emotions anger, disgust, fear, sad, and surprise has been combined with the AffectNet data.</p>

## Results
<p>The model is able to achieve an accuracy of approximately 80%.</p>

## Instructions
### Training the Model
<p>
<ol>
<li> Download the datasets and store them in a folder named 'data' with the datasets being stored in 'affectnet' and 'fer' folder inside it respectively. 
<li> Run the affectnetEE.py file to train the model. A new 'models' folder would be formed containing the model json file and the weights file.
</ol></p> 
<h3> Running the Website </h3>
<p>Run the process.py file and open localhost:5001 on your browser. Allow the browser to access the webcam. The emotion detected and the engagement status would be viewed on the website.</p>
