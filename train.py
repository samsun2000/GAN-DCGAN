import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import os

from utils import get_EMNIST
from dcgan import weights_init, Generator, Discriminator

# Random seeds to ensure reproducibility.
seed = 369
random.seed(seed)
torch.manual_seed(seed)
print("Random Seed: ", seed)

# Parameters to define for the program
#TODO!!! Change the parameters according to your own needs
params = {
    "bsize" : 32,# Batch size during training.
    'imsize' : 32,# Spatial size of training images. All images will be resized to this size during preprocessing.
    'nc' : 3,# Number of channles in the training images. For greyscale images this is 3.
    'nz' : 100,# Size of the Z latent vector (the input to the generator).
    'ngf' : 64,# Size of feature maps in the generator. The depth will be multiples of this.
    'ndf' : 64, # Size of features maps in the discriminator. The depth will be multiples of this.
    'nepochs' : 100,# Number of training epochs.
    'lr' : 0.0001,# Learning rate for optimizers
    'beta1' : 0.5,# Beta1 hyperparam for Adam optimizer
    'save_epoch' : 10,
    'n_classes' : 2}# Save step.

# Use GPU if cuda is available, else use CPU.
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
#device = torch.device("cpu")
#Use the above code if GPU is not available

# Get the data.
dataloader = get_EMNIST(params)
num_epochs = params['nepochs']

def generate_and_plot(netG, noise, labels, fig_size=(5, 5)):
    # Set the generator in evaluation mode
    netG.eval()

    # Generate images
    with torch.no_grad():
        gen_images = netG(noise, labels)

    # Plot the generated images
    plot_result(netG, noise, labels, -1, save=True, fig_size=fig_size, show=True)

#The help function to plot the loss trend
def plot_loss(d_losses, g_losses, num_epoch, save=False, save_dir='DCGAN_results/', show=False):
    fig, ax = plt.subplots()
    ax.set_xlim(0, num_epochs)
    ax.set_ylim(0, max(np.max(g_losses), np.max(d_losses))*1.1)
    plt.xlabel('Epoch {0}'.format(num_epoch + 1))
    plt.ylabel('Loss values')
    plt.plot(d_losses, label='Discriminator')
    plt.plot(g_losses, label='Generator')
    plt.legend()

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + 'DCGAN_losses_epoch_{:d}'.format(num_epoch + 1) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()

#The help function for plotting results
def plot_result(generator, noise, label, num_epoch, save=False, save_dir='DCGAN_results/', show=True, fig_size=(5, 5)):
    #TODO!!! Call the model G (generator) properly to generate images from noises (Part 3.3)
    gen_image = generator(noise, label)  # Generate images from noise and label
    gen_image_np = gen_image.detach().cpu().numpy()
    unique_images = np.unique(gen_image_np, axis=0)
    print("Number of unique images:", len(unique_images))
    generator.train()

    n_rows = np.sqrt(noise.size()[0]).astype(np.int32)
    n_cols = np.sqrt(noise.size()[0]).astype(np.int32)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
    for ax, img in zip(axes.flatten(), gen_image):
        ax.axis('off')
        ax.adjustable = 'box-forced'
        img = (((img - img.min()) * 255) / (img.max() - img.min())).cpu().data.numpy().transpose(1, 2, 0).astype(
            np.uint8)
        ax.imshow(img.astype(np.uint8), cmap='Greys_r')
    plt.subplots_adjust(wspace=0, hspace=0)
    title = 'Epoch {0}'.format(num_epoch + 1)
    fig.text(0.5, 0.04, title, ha='center')

    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + 'DCGAN_epoch_{:d}'.format(num_epoch + 1) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()

netG = Generator(params).to(device)
# The weights_init() function is called to randomly initialize all weights to mean=0.0, stddev=0.2
netG.apply(weights_init)
# Print the model to check the structure
print(netG)

netD = Discriminator(params).to(device)
# Apply the weights_init() function to randomly initialize all
# weights to mean=0.0, stddev=0.2
netD.apply(weights_init)
# Print the model.
print(netD)

# For defining the loss function to use: Binary Cross Entropy loss function.
criterion = nn.BCELoss()

#Define a fixed noise map to ensure the reproducibility of your results
fixed_noise = torch.randn(64, params['nz'], 1, 1, device=device)
fixed_labels = torch.randint(0, 2, (64,), dtype=torch.long, device=device)

model_path = 'netG_epoch_10.pth'  # or any other saved model path like 'netG_epoch_10.pth'
netG.load_state_dict(torch.load(model_path))

# Assuming you want to generate and plot images using the fixed noise and labels
generate_and_plot(netG, fixed_noise, fixed_labels)

real_label = 1
fake_label = 0

# Optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))

# Stores generated images as training progresses.
img_list = []
# To Store G's and D's losses during training.
G_losses = []
D_losses = []



iters = 0

print("Now Training...")
print("-"*25)

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        # Transfer data tensor to GPU/CPU (device)
        real_data, real_labels = data[0].to(device), data[1].to(device)
        b_size = real_data.size(0)
        # Set real labels
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)

        ## (a) Train D with real data ##
        netD.zero_grad()
        output = netD(real_data,real_labels).view(-1).float()
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        ## (b) Train D with fake data ##
        # Generate fake images with G
        noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
        rand_labels = torch.randint(0, 2, (b_size,), dtype=torch.long, device=device)
        fake_data = netG(noise, rand_labels)
        # Set fake labels
        label.fill_(fake_label)
        output = netD(fake_data.detach(), rand_labels).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # Combine losses and update D
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        # For the generator, we want to classify the fake data as real
        label.fill_(real_label)
        output = netD(fake_data,rand_labels).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()

        # Check progress of training
        if i % 10 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, params['nepochs'], i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save the losses for plotting
        G_losses.append(errG.item())
        D_losses.append(errD.item())

    # Save the model
    if epoch % params['save_epoch'] == 0:
        torch.save(netG.state_dict(), f'netG_epoch_{epoch}.pth')
        torch.save(netD.state_dict(), f'netD_epoch_{epoch}.pth')
        plot_loss(D_losses, G_losses, epoch, save=True)

    #Show result for fixed noise
    if epoch % 10 == 0:
        plot_result(netG, fixed_noise,fixed_labels, epoch, save=True, fig_size=(5, 5))

# Save the final trained model
torch.save(netG.state_dict(), 'netG_final.pth')
torch.save(netD.state_dict(), 'netD_final.pth')

# Plot the training losses
plot_loss(D_losses, G_losses, num_epochs, save=True, show=True)

plot_result(netG, fixed_noise,fixed_labels, num_epochs, save=True, fig_size=(10, 10))


# Animation showing the improvements of the generator.


#TODO!!! Part 4 (Challenge) Make necessary changes to adapt to Conditional DCGAN model.
#Any reasonable attempts you made could get some points, even the new model doesn't work properly.

#This is normal DCGAN version for single class character
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import matplotlib.pyplot as plt
# import random
# import os
#
# from utils import get_EMNIST
# from dcgan import weights_init, Generator, Discriminator
#
# # Random seeds to ensure reproducibility.
# seed = 369
# random.seed(seed)
# torch.manual_seed(seed)
# print("Random Seed: ", seed)
#
# # Parameters to define for the program
# #TODO!!! Change the parameters according to your own needs
# params = {
#     "bsize" : 32,# Batch size during training.
#     'imsize' : 32,# Spatial size of training images. All images will be resized to this size during preprocessing.
#     'nc' : 3,# Number of channles in the training images. For greyscale images this is 3.
#     'nz' : 100,# Size of the Z latent vector (the input to the generator).
#     'ngf' : 64,# Size of feature maps in the generator. The depth will be multiples of this.
#     'ndf' : 64, # Size of features maps in the discriminator. The depth will be multiples of this.
#     'nepochs' : 100,# Number of training epochs.
#     'lr' : 0.0001,# Learning rate for optimizers
#     'beta1' : 0.5,# Beta1 hyperparam for Adam optimizer
#     'save_epoch' : 10}# Save step.
#
# # Use GPU if cuda is available, else use CPU.
# device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# #device = torch.device("cpu")
# #Use the above code if GPU is not available
#
# # Get the data.
# dataloader = get_EMNIST(params)
# num_epochs = params['nepochs']
#
# #The help function to plot the loss trend
# def plot_loss(d_losses, g_losses, num_epoch, save=False, save_dir='DCGAN_results/', show=False):
#     fig, ax = plt.subplots()
#     ax.set_xlim(0, num_epochs)
#     ax.set_ylim(0, max(np.max(g_losses), np.max(d_losses))*1.1)
#     plt.xlabel('Epoch {0}'.format(num_epoch + 1))
#     plt.ylabel('Loss values')
#     plt.plot(d_losses, label='Discriminator')
#     plt.plot(g_losses, label='Generator')
#     plt.legend()
#
#     # save figure
#     if save:
#         if not os.path.exists(save_dir):
#             os.mkdir(save_dir)
#         save_fn = save_dir + 'DCGAN_losses_epoch_{:d}'.format(num_epoch + 1) + '.png'
#         plt.savefig(save_fn)
#
#     if show:
#         plt.show()
#     else:
#         plt.close()
#
# #The help function for plotting results
# def plot_result(generator, noise, num_epoch, save=False, save_dir='DCGAN_results/', show=True, fig_size=(5, 5)):
#     #TODO!!! Call the model G (generator) properly to generate images from noises (Part 3.3)
#     gen_image = generator(noise)  # 从噪声生成图像
#     generator.train()
#     #Important! If you are calling this plot result function during training,
#     # don't forget to change the model mode from eval() to train().
#
#     n_rows = np.sqrt(noise.size()[0]).astype(np.int32)
#     n_cols = np.sqrt(noise.size()[0]).astype(np.int32)
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
#     for ax, img in zip(axes.flatten(), gen_image):
#         ax.axis('off')
#         ax.adjustable = 'box-forced'
#         img = (((img - img.min()) * 255) / (img.max() - img.min())).cpu().data.numpy().transpose(1, 2, 0).astype(np.uint8)
#         ax.imshow(img.astype(np.uint8), cmap='Greys_r') # The default grey scale visualization is reversed. Using "_r" to show the correct scale
#     plt.subplots_adjust(wspace=0, hspace=0)
#     title = 'Epoch {0}'.format(num_epoch+1)
#     fig.text(0.5, 0.04, title, ha='center')
#
#     # save figure
#     if save:
#         if not os.path.exists(save_dir):
#             os.mkdir(save_dir)
#         save_fn = save_dir + 'DCGAN_epoch_{:d}'.format(num_epoch+1) + '.png'
#         plt.savefig(save_fn)
#
#     if show:
#         plt.show()
#     else:
#         plt.close()
#
# netG = Generator(params).to(device)
# # The weights_init() function is called to randomly initialize all weights to mean=0.0, stddev=0.2
# netG.apply(weights_init)
# # Print the model to check the structure
# print(netG)
#
# netD = Discriminator(params).to(device)
# # Apply the weights_init() function to randomly initialize all
# # weights to mean=0.0, stddev=0.2
# netD.apply(weights_init)
# # Print the model.
# print(netD)
#
# # For defining the loss function to use: Binary Cross Entropy loss function.
# criterion = nn.BCELoss()
#
# #Define a fixed noise map to ensure the reproducibility of your results
# fixed_noise = torch.randn(64, params['nz'], 1, 1, device=device)
#
# real_label = 1
# fake_label = 0
#
# # Optimizers for both G and D
# optimizerD = optim.Adam(netD.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
# optimizerG = optim.Adam(netG.parameters(), lr=params['lr'], betas=(params['beta1'], 0.999))
#
# # Stores generated images as training progresses.
# img_list = []
# # To Store G's and D's losses during training.
# G_losses = []
# D_losses = []
#
# iters = 0
#
# print("Now Training...")
# print("-"*25)
#
# for epoch in range(num_epochs):
#     for i, data in enumerate(dataloader, 0):
#
#         ############################
#         # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
#         ###########################
#
#         # Transfer data tensor to GPU/CPU (device)
#         real_data = data[0].to(device)
#         b_size = real_data.size(0)
#         # Set real labels
#         label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
#
#         ## (a) Train D with real data ##
#         netD.zero_grad()
#         output = netD(real_data).view(-1).float()
#         errD_real = criterion(output, label)
#         errD_real.backward()
#         D_x = output.mean().item()
#
#         ## (b) Train D with fake data ##
#         # Generate fake images with G
#         noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
#         fake_data = netG(noise)
#         # Set fake labels
#         label.fill_(fake_label)
#         output = netD(fake_data.detach()).view(-1)
#         errD_fake = criterion(output, label)
#         errD_fake.backward()
#         D_G_z1 = output.mean().item()
#
#         # Combine losses and update D
#         errD = errD_real + errD_fake
#         optimizerD.step()
#
#         ############################
#         # (2) Update G network: maximize log(D(G(z)))
#         ###########################
#         netG.zero_grad()
#         # For the generator, we want to classify the fake data as real
#         label.fill_(real_label)
#         output = netD(fake_data).view(-1)
#         errG = criterion(output, label)
#         errG.backward()
#         D_G_z2 = output.mean().item()
#         optimizerG.step()
#
#         # Check progress of training
#         if i % 10 == 0:
#             print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
#                   % (epoch, params['nepochs'], i, len(dataloader),
#                      errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
#
#         # Save the losses for plotting
#         G_losses.append(errG.item())
#         D_losses.append(errD.item())
#
#     # Save the model
#     if epoch % params['save_epoch'] == 0:
#         torch.save(netG.state_dict(), f'netG_epoch_{epoch}.pth')
#         torch.save(netD.state_dict(), f'netD_epoch_{epoch}.pth')
#         plot_loss(D_losses, G_losses, epoch, save=True)
#
#     #Show result for fixed noise
#     if epoch % 10 == 0:
#         plot_result(netG, fixed_noise, epoch, save=True, fig_size=(5, 5))
#
# # Save the final trained model
# torch.save(netG.state_dict(), 'netG_final.pth')
# torch.save(netD.state_dict(), 'netD_final.pth')
#
# # Plot the training losses
# plot_loss(D_losses, G_losses, num_epochs, save=True, show=True)
#
# plot_result(netG, fixed_noise, num_epochs, save=True, fig_size=(10, 10))
#
#
# # Animation showing the improvements of the generator.
#
#
# #TODO!!! Part 4 (Challenge) Make necessary changes to adapt to Conditional DCGAN model.
# #Any reasonable attempts you made could get some points, even the new model doesn't work properly.