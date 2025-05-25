# ANIME FACE GENERATOR:

This uses a DCGAN (Deep Convolutional General Adversary Network) to create anime faces. It is trained using the animeface dataset and generates images based on random noise vectors.

A Generative Adversarial Network (GAN) consist of two networks — a **Generator** and a **Discriminator** — that compete with each other. The generator tries to create realistic data, while the discriminator tries to distinguish real data from generated ones.This project uses **DCGAN**, a type of GAN that works especially well with image data, to generate 64x64 pixel anime face images. Over time, the generator learns the distribution of anime faces and starts producing realistic-looking characters.

## How this works: 

DCGAN is a type of GAN that uses convolutional layers (like in CNNs) instead of fully connected layers, which makes it very effective for images.
###  Generator:
Input: A random noise vector z (e.g., 100-dim normal distribution).
Goal: Transform this vector into a realistic image.
#### Layers:
Linear layer → reshaped into a small feature map
Several ConvTranspose2d layers (upsampling) with BatchNorm and ReLU activations.
Final layer uses Tanh() to scale pixel values to [-1, 1] (matches preprocessed image scale)

### Discriminator:
Input: An image (either real or fake)
Goal: Predict whether it’s real (1) or fake (0)
#### Layers: 
Several Conv2d layers (downsampling) with BatchNorm and LeakyReLU activations
Final layer outputs a single number (probability it's real), using Sigmoid()

### Training Loop
Training a DCGAN involves alternating updates to D and G using backpropagation:
#### 1) Train the Discriminator
-Sample a batch of real images from the dataset.
-Sample a batch of fake images by feeding random noise to the Generator.
-Let the Discriminator classify both:
-D(real images) → 1 (real) and D(fake images) → 0 (fake)
-Compute loss: Loss_D = BCE(D(real), 1) + BCE(D(fake), 0)
-Backpropagate and update only D's weights.

#### 2) Train the Generator
Generate a new batch of fake images from noise.
Pass these to the Discriminator.
Now, we want the Generator to fool the Discriminator -> D(fake images) → 1 (even though they’re fake)
Compute loss: Loss_G = BCE(D(fake), 1)
Backpropagate and update only G's weights.





## Dataset link: 
 https://www.kaggle.com/datasets/splcher/animefacedataset



## Streamlit App Link: 
https://anime-face-generator-rohith.streamlit.app

