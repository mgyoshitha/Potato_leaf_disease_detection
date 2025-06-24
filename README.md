Potato Leaf Disease Detection using Deep Learning(using Streamlit)
A deep learning–based solution for automatically detecting diseases in potato leaves from images — helping farmers and agronomists identify and manage plant health effectively.

About the Project
Potatoes are vulnerable to a number of diseases, which can significantly reduce crop yield and quality. This project uses a Convolutional Neural Network (CNN) to classify images of potato leaves into one of the following categories:
Healthy
Early Blight
Late Blight
This tool can assist in rapid field-level disease diagnosis using images.

Dataset
It contains images categorized into:
Potato___Healthy
Potato___Early_blight
Potato___Late_blight
Each image is labeled, preprocessed (resized, normalized), and split into train, validation, and test sets.

Model Architecture
Input: Leaf images resized to 128x128 or 224x224 pixels
Model Type: CNN
Layers:
Convolution + ReLU
Max Pooling
Dropout
Dense layers
Output: Softmax classifier with 3 classes

Technologies Used
Python
TensorFlow 
NumPy, Pandas, Matplotlib
OpenCV (for image preprocessing)
Jupyter Notebook
Streamlit

Results
Metric	Value
Accuracy	98.5%
Precision	98.2%
Recall	98.4%
F1-Score	98.3%
Confusion matrix and sample predictions are available in the notebook.

