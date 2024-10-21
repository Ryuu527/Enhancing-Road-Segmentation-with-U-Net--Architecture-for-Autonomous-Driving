import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

# Set paths
base_path = '/RoadSegmentationDataset_TrainingData'
image_path = os.path.join(base_path, 'images')
mask_path = os.path.join(base_path, 'masks')

# Load and process images and masks
def load_images_and_masks(image_dir, mask_dir, target_size=(512, 512)):
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')])
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])

    images = []
    masks = []

    for file in image_files:
        img = cv2.imread(file, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)  # Resize image
        images.append(img / 255.0)  # Normalize images

    for file in mask_files:
        mask = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, target_size)  # Resize mask
        masks.append(mask / 255.0)  # Normalize masks

    return np.array(images), np.expand_dims(masks, -1)

images, masks = load_images_and_masks(image_path, mask_path)

# Split data
X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

# Create U-Net model
def create_unet_model(input_shape=(512, 512, 3)):
    inputs = Input(input_shape)
    # Encoder
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    # Decoder
    u6 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(p1)
    u6 = concatenate([u6, c1])
    c6 = Conv2D(32, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(c6)
    model = Model(inputs, outputs=c6)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = create_unet_model()
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=16, epochs=50)

# Visualization of results
def display_results(images, true_masks, predicted_masks):
    num_samples_to_display = len(images)
    fig, axes = plt.subplots(num_samples_to_display, 3, figsize=(15, 5 * num_samples_to_display))

    for i in range(num_samples_to_display):
        # Original Image
        ax = axes[i, 0]
        ax.imshow(images[i])
        ax.set_title('Original Image')
        ax.axis('off')

        # Original Mask
        ax = axes[i, 1]
        ax.imshow(true_masks[i].squeeze(), cmap='jet')
        ax.set_title('Original Mask')
        ax.axis('off')

        # Predicted Mask
        ax = axes[i, 2]
        ax.imshow(predicted_masks[i].squeeze(), cmap='jet', vmin=0, vmax=1)  # Set color map limits
        ax.set_title('Predicted Mask')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage with your model's predictions
num_samples_to_display = 3  # Set how many samples you want to display
indices = np.random.choice(range(len(X_test)), num_samples_to_display, replace=False)
sample_images = [X_test[i] for i in indices]
sample_true_masks = [y_test[i] for i in indices]
sample_predicted_masks = [model.predict(np.expand_dims(X_test[i], axis=0))[0] for i in indices]

display_results(sample_images, sample_true_masks, sample_predicted_masks)
