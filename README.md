# ANIME FACE GENERATOR:

This uses a DCGAN (Deep Convolutional General Adversary Network) to create anime faces. It is trained using the animeface dataset and generates images based on random noise vectors.

A Generative Adversarial Network (GAN) consist of two networks — a **Generator** and a **Discriminator** — that compete with each other. The generator tries to create realistic data, while the discriminator tries to distinguish real data from generated ones.This project uses **DCGAN**, a type of GAN that works especially well with image data, to generate 64x64 pixel anime face images. Over time, the generator learns the distribution of anime faces and starts producing realistic-looking characters.

## How this works: 

 **Generator** takes a random noise vector (latent vector) and transforms it through transposed convolutions into a fake image.
 **Discriminator** tries to classify images as real or fake.
   Both models train in a loop — the generator improves at fooling the discriminator, and the discriminator improves at spotting fakes.

Eventually, the generator learns to produce images that are indistinguishable from real anime faces (at least to the discriminator).

## Dataset link: 
 https://www.kaggle.com/datasets/splcher/animefacedataset



## Streamlit App Link: 
https://anime-face-generator-rohith.streamlit.app

