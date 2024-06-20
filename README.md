# GAN-DCGAN
In this implementation, we have developed a DeepConvolutional Generative Adversarial Network (DCGAN) and incorporated features of Conditional GAN, enabling the generation of specific images under designated conditions. This means the produced images are not purely random but are generated based on provided labels or descriptions. 

By constructing and fine-tuning both the generator and discriminator neural networks, this model aims to produce images that are indistinguishable from real samples, showcasing the prowess of GANs in image synthesis. The
introduction of the Conditional GAN adds a layer of flexibility and broadens the application spectrum. Through our iterative training process, the generator has progressively enhanced its capability to craft realistic images, successfully highlighting the potential of DCGANs and Conditional GANs in applications ranging from image generation to advanced uses such as data augmentation and anomaly detection.

How to run:
to run conditional DCGAN:
Direct to root DCGAN-framework， then run python train.py

To load trained model:
modify line 129 in train.py to set the model then run python train.py
the first image output will be generated image from trained model.

The normal single class character of dcgan.py and train.py are commented at the end.
