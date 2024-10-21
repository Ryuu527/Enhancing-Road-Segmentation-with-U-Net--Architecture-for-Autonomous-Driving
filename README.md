Project: Road Segmentation for Autonomous Driving

Description
This project implements a U-Net based model for semantic segmentation of roads from satellite images. The purpose is to enhance autonomous driving systems by improving their ability to detect and differentiate roadways from surrounding environments.

Installation

Prerequisites
Python 3.8 or later
TensorFlow 2.x
OpenCV
NumPy
Matplotlib

Setup
Clone this repository to your local machine.
Ensure you have the required packages installed:

pip install tensorflow opencv-python numpy matplotlib


Usage

Running the Script
To run the segmentation model, navigate to the project directory and execute the main script:

python script.py

Data
The images folder should contain the images used for training and testing the model, while the masks folder should contain the corresponding ground truth segmentation masks. Ensure each image in the images folder has a corresponding mask in the masks folder.

Results
Results are displayed on-screen as the model processes each image. Additionally, results are saved in the results directory, which is automatically created if it does not exist.

Contributing
Contributions to this project are welcome. Please fork the repository and submit a pull request with your improvements.

License
This project is licensed under the MIT License - see the LICENSE.md file for details.

Contact
Name: Bong Hong Jun
Email: 22018519@imiail.sunway.edu.my
Organization: School of Engineering and Technology, Sunway University, Selangor, Malaysia