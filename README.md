# Software Documentation

## Preamble

Originally, our intended software aimed to capture images of biosensors, count the sodium alginate hydrogels present, and identify any color changes in each hydrogel. The count of hydrogels exhibiting color changes would then serve as the basis for calculating probabilities to be integrated into our mapping software. However, due to limitations in training data, we were unable to develop the full-fledged software as initially planned.

Instead, we have created a proof of concept color detection software specifically designed to identify color changes in a single sodium alginate hydrogel. Our current focus is to demonstrate how the automated system was intended to function, showcasing its capabilities within these constraints. We hope to expand upon this software in the future to include the full functionality of our original software.

## How the intended software was supposed to work

The software was supposed to accept an image as input.

![Image](images/hydrogel.jpeg)
*Figure 0: Sample image of a sodium alginate hydrogels*

The software would then count the number of hydrogels present in the image and identify any color changes in each hydrogel. For example the image above has ~95 hydrogels. Our bio sensor have bioindicators meants to react in the presence of both lithium and arsenic (a lithium pathfinder). The color change is meant to be a visual indicator of the presence of lithium. If the hydrogel has turned to red it means that lithium is present. While as if the hydrogel has turned to yellow it means that arsenic is present. If the hydrogel has not changed color it means that neither lithium nor arsenic is present.

If we have both red and yellow hydrogels then there is a high chance of lithium being present. If we have only red hydrogels then there is a medium chance of lithium being present. If we have only yellow hydrogels then there is a low chance of lithium being present. If we have no red or yellow hydrogels then there is no chance of lithium being present.

The probability of lithium being present is calculated using the following formula:

$$P(Lithium) = \frac{0.5 \cdot (number\_of\_red\_hydrogels + number\_of\_yellow\_hydrogels)}{total\_number\_of\_hydrogels}$$

Below is an explanation of the color detection software that we have developed.

## Introduction
The Color Detection Software is a tool developed for our project aimed at detecting the color of bioindicators based on the RGB value of an image. The software utilizes two machine learning models: Decision Trees and K-Nearest Neighbors (KNN). This documentation provides an overview of the software, its functionality, and the process of developing and improving it.

![Image](images/training_dataset.png)
*Figure 1: Training Dataset*

## Table of Contents
1. **Getting Started**
    - Prerequisites
    - Installation
    - Use the model
2. **Data Collection**
    - Folder Structure
    - Image Selection
3. **Data Preprocessing**
    - Extracting RGB Values
    - Data Labeling
4. **Machine Learning Models**
    - Decision Tree Classifier
    - Model Evaluation
    - K-Nearest Neighbors Classifier
5. **Model Comparison**
    - Performance Metrics
    - Accuracy Visualization
6. **Model Deployment**
    - Saving the Model
7. **Conclusion**
    - Project Summary
    - Future Improvements

## 1. Getting Started
### 1.1 Prerequisites
- Python 3.x
- OpenCV (cv2)
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

### 1.2 Installation
Install the required libraries using pip:
```bash
pip install opencv-python pandas numpy matplotlib scikit-learn
```

### 1.3 Use the model
To use the model, load the `color_detection_model.pkl` file using the Pickle library:
```python
import pickle

model = pickle.load(open('color_detection_model.pkl', 'rb'))
```
Then, use the `predict` function to predict the color of an image:
```python
model.predict([[255, 255, 0]])
# Output: array([6])
```
Or read the RGB values using the `cv2` (python-opencv) library

```python
import cv2

path = cv2.imread('image.jpg')
image = cv2.cvtColor(path, cv2.COLOR_BGR2RGB)
rgb = image[0][0]
model.predict([rgb])
```

The model returns a numerical value representing the color category. The color categories are as follows:

#
- 0: Grey
- 1: Black
- 2: White
- 3: Orange
- 4: Brown
- 5: Blue
- 6: Yellow
- 7: Green
- 8: Violet
- 9: Red

## 2. Data Collection
### 2.1 Folder Structure
The software expects a structured dataset located in a folder called 'training_dataset.' Inside this folder, each subfolder represents a different color category, and it contains images of that color.

### 2.2 Image Selection
For higher training accuracy, the number of images under each color category should be the same or roughly the same.

## 3. Data Preprocessing
### 3.1 Extracting RGB Values
The software extracts the RGB values from each image using the cv2 library. The RGB values are stored in a Pandas DataFrame.

### 3.2 Data Labeling
The color categories are labeled numerically using the `factorize` function from Pandas.

## 4. Machine Learning Models
### 4.1 Decision Tree Classifier
A Decision Tree Classifier is trained on the dataset with different `max_depth` values to determine the optimal tree depth. The accuracy of the model is evaluated using a test dataset.

### 4.2 K-Nearest Neighbors Classifier
A K-Nearest Neighbors (KNN) Classifier is trained on the dataset with varying numbers of neighbors. The 'distance' metric and 'euclidean' distance measure are used. The accuracy of the KNN model is also evaluated.

## 5. Model Comparison
### 5.1 Performance Metrics
Both Decision Tree and KNN models are evaluated using the accuracy metric, which measures the proportion of correctly classified samples.

### 5.2 Accuracy Visualization
The software generates accuracy vs. parameter plots to visualize the performance of both models and help determine the best model for the task.

![Image](images/accuracy_dtree.png)
*Figure 2: Decision Tree Accuracy Plot*

![Image](images/accuracy_knn.png)
*Figure 3: KNN Accuracy Plot*

## 6. Model Deployment
The final KNN model with the highest accuracy is saved using the Pickle library as 'color_detection_model.pkl.'

## 7. Conclusion
After exploring two machine learning models, Decision Trees and K-Nearest Neighbors, the KNN model achieved the highest accuracy (92%) and was chosen for deployment. Future improvements could include fine-tuning the model and expanding the dataset for more accurate detection.

This software serves as a valuable tool for color-based bioindicator detection and can be further enhanced to support a wider range of applications.

