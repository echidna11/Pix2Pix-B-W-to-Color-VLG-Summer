# Pix2Pix-B-W-to-Color-VLG-Summer
VLG Summer Project 2022

## Usage

- Open the pix2pix jupyter notebook on Google Colab.
- Select GPU in runtime Type and then hit connect.
- Run the first 2 Commands downloading the dataset into root or you can go forward with the dataset of your choice making changes accordingly.
- within the directory of the dataset named coco_sample, create an Output and a train folder, Output, as the name suggests will store the output and train will store the B/W images that come from the RGB images present in our dataset.
- Once that is done you can run the rest of the commands and start training your model.
- The result after training ~4000 images for 25 epochs was descent enough(images present below) and also helped out with time constraints.


## Description
Pix2Pix is a Generative Adversarial Network, or GAN, model designed for general purpose image-to-image translation.
Within this Project, I've implemented a GAN model that is able to colorize black and white images with a fair amount of accuracy.
The discriminator used in the pix2pix paper is a PatchGAN discriminator model which is designed based on the receptive field or the effective receptive field.
A PatchGAN with size 70x70 is used as the paper states "The 70×70 PatchGAN […] achieves slightly better scores. Scaling beyond this, to the full 286×286 ImageGAN, does not appear to improve the visual quality of the results.". This essentially means that the output of the model maps to a 70×70 square of the input image. The config is defined as a set of Convolution-BatchNorm-LeakyReLU layers. The discriminator model will classify whether the 70x70 patches of the input image as real/fake.

The U-Net Generator model specified in the paper has been used. It is an encoder-decoder model specific for image translation. Within this architecture the input is passed through a series of layers that progressively downsample, until a bottleneck layer, at which point the process is reversed.
![Architecture-of-the-U-Net-Generator-Model jpg](https://user-images.githubusercontent.com/76242511/176702599-3c6b25b3-7032-4204-a6e8-e32f31536802.jpg)

Here in, we add skip connections between each layer i and layer n − i, where n is the total number of layers. Each skip connection concatenates all channels at layer i with those at layer n − i.

The weights of the generator will be updated via both adversarial loss via the discriminator output and L1 loss via the direct image output. The loss scores are added together, where the L1 loss is treated as a regularizing term and weighted via a hyperparameter called lambda.

Below are a few generated outputs.

## Generated Outputs(Results):
Real Images were not a part of the dataset and were captured in grayscale originally:

![Screenshot (244)](https://user-images.githubusercontent.com/76242511/176699376-0222e55d-7cb6-4b54-9e62-06fdd5222bfd.png)
