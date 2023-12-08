# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import os
from time import perf_counter
import itertools

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import dnnlib
import legacy

import utils_vis
import utils_data

VERBOSE = True

#----------------------------------------------------------------------------

def get_disc_features(D,
                      img):
    """
    This function extracts discrimator's features for the image.

    : param D: trained discriminator
    : param img: image for which the features are desired

    : return : extracted features
    """
    feats = None
    for res in D.block_resolutions:
        feats, img = getattr(D, f'b{res}')(feats, img)
    return feats

#----------------------------------------------------------------------------

def save_real_and_generated_samples(G,
                                    nsamples,
                                    fname,
                                    device):
    """
    This function saves real and generated images.
    : param G : Trained Generator
    : param nsamples : Number of images to generate
    : param fname : path where the real and generated images should be saved
    : param device : device on which to run the generation
    """
    # save real images
    x_samples = utils_data.load_real_images(nsamples)
    utils_vis.save_images(x_samples, f"{fname}_real.png")
    # save generated images
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore
    z_samples = np.random.RandomState(123).randn(nsamples, G.z_dim)
    x_samples = G(torch.from_numpy(z_samples).to(device), None)
    utils_vis.save_images(x_samples.detach().cpu().numpy(), f"{fname}_fake.png")
    # return
    return 0

#----------------------------------------------------------------------------

def load_image(fname,
               size):
    """
    This function loads an image from disk.
    : param fname: The filename of the image to be loaded.
    : param size: Size of the output image.
    : return : image as a numpy array of size
               (size, 1) for grayscale images /
               (size, 3) for rgb images
    """
    image_pil = PIL.Image.open(fname).convert('L') # .convert('RGB') for rgb images
    # crop image if it not square
    width, height = image_pil.size
    short = min(width, height)
    image_pil = image_pil.crop(((width - short) // 2,
                                (height - short) // 2,
                                (width + short) // 2,
                                (height + short) // 2))
    # resize to required resolution
    image_pil = image_pil.resize(size, PIL.Image.Resampling.LANCZOS)
    # expand_dims needed for grayscale images
    return np.expand_dims(np.array(image_pil, dtype=np.uint8), axis=-1)

#----------------------------------------------------------------------------

def load_nets(network_pkl,
              device):
    """
    This function load networks from a pickle file.

    : param network_pkl: Path of the pickle file
    : param device: Device on which to define the models

    : return gen: Generator
    : return disc: Discriminator
    """
    print(f'Loading networks from {network_pkl}...')
    with dnnlib.util.open_url(network_pkl) as file_pointer:
        data = legacy.load_network_pkl(file_pointer)
    G = data['G_ema'].requires_grad_(False).to(device) # type: ignore
    D = data['D'].requires_grad_(False).to(device) # type: ignore
    return G, D

#----------------------------------------------------------------------------

def save_video_showing_optimization(fname,
                                    w_steps,
                                    m_steps,
                                    G,
                                    losses,
                                    target,
                                    step_delta = 10):
    """
    This function saves a video showing the optimization process.
    """
    
    # create video file
    video = imageio.get_writer(fname, mode='I', fps=1, codec='libx264', bitrate='16M')
    print (f'Saving optimization progress video at {fname}.')

    # convert target from uint to float between -1 and 1
    target_ = target.astype(np.float32) / 127.5 - 1 # range [-1, 1]

    # save video frames
    for step, wm_tuple in enumerate(zip(w_steps, m_steps)):
        if step % step_delta == 0:

            w_vector, mask = wm_tuple

            # synthesize image for this w_vector
            synth_image = G.synthesis(w_vector.unsqueeze(0), noise_mode='const') # range [-1, 1]
            synth_image = synth_image.permute(0, 2, 3, 1)[0].cpu().numpy() # shape [nx, ny]

            # error image
            error_image = abs(synth_image - target_) / 2.0

            # estimated mask at this iteration
            mask = mask.cpu().numpy() # range [0, 1]

            # get image containing line plots for the losses until this step
            # this takes quite long. do it only once for now
            # pass final step instead of current step to save time.
            if step == 0:
                loss_plot = utils_vis.plot_losses_and_return_image(losses,
                                                                   step=w_steps.shape[0]-1, 
                                                                   size=(mask.shape[1:]))

            # prepare video frame
            images = [target_, synth_image, error_image, mask, loss_plot]
            cmaps = ['gray'] * (len(images) - 1) + [None]
            clims = [(-1.0, 1.0)] * 2 + [(0.0, 1.0), (0.0, 1.0), (None)]
            video_frame = utils_vis.prepare_video_frame(data = zip(images, cmaps, clims))

            # append to video
            video.append_data(video_frame)                     
            
    # close video
    video.close()

#----------------------------------------------------------------------------

def logprint(*args):
    """
    This function prints its inputs.
    """
    if VERBOSE:
        print(*args)
    # return
    return 0

#----------------------------------------------------------------------------

def get_w_stats(num_samples,
                G,
                device):
    """
    This function computes mean and std. dev in the w-space.
    """
    logprint(f'Computing W midpoint and stddev using {num_samples} samples...')
    z_samples = np.random.RandomState(123).randn(num_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / num_samples) ** 0.5
    return w_avg, w_std

#----------------------------------------------------------------------------

def get_lr(normalized_step,
           lr_rampdown_length,
           lr_rampup_length,
           initial_learning_rate):
    """
    This function computes the learning rate at the given step in the optimization.
    """
    lr_ramp = min(1.0, (1.0 - normalized_step) / lr_rampdown_length)
    lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
    lr_ramp = lr_ramp * min(1.0, normalized_step / lr_rampup_length)
    return initial_learning_rate * lr_ramp

#----------------------------------------------------------------------------

def get_w_noise_scale(normalized_step,
                      w_std,
                      initial_noise_factor,
                      noise_ramp_length):
    """
    This function computes the scale of the noise
    to be added to the w-vector at the given step in the optimization.
    """
    return w_std * initial_noise_factor * max(0.0, 1.0 - normalized_step / noise_ramp_length) ** 2

#----------------------------------------------------------------------------

def get_noise_reg_loss(noise_bufs):
    """
    This function computes the noise regularization loss described in styleGANv2.
    """
    reg_loss = 0.0
    for val in noise_bufs.values():
        noise = val[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
        while True:
            reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
            reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
            if noise.shape[2] <= 8:
                break
            noise = F.avg_pool2d(noise, kernel_size=2)
    return reg_loss

#----------------------------------------------------------------------------

def project(
    G,
    D,
    target: torch.Tensor, # [C,H,W], dynamic range [0,255], W, H must match G res
    *,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    lambda_n                   = 1e5,
    lambda_d                   = 0.01,
    lambda_m                   = 1.0,
    device: torch.device
):
    """
    This function projects a target image onto the latent space of the GAN.
    
    : param G: trained Generator
    : param D: trained Discriminator

    : return w: w vectors over the optimization trajectory.
    : return mask: masks over the optimization trajectory.
    : return losses: dict with keys loss_types and values lists of size [num_steps].

    """
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore
    D = copy.deepcopy(D).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w stats.
    w_avg, w_std = get_w_stats(num_samples = w_avg_samples, G = G, device = device)

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    # Features for target image.
    # Scale intensities to -1 to 1 # [1, nc, nx, ny]
    target_images = target.unsqueeze(0).to(device).to(torch.float32) / 127.5 - 1
    target_features = get_disc_features(D, target_images)

    # Initialize mask: same size as the target image
    mask = torch.zeros_like(input = target_images, requires_grad=True) # [1, nc, nx, ny]

    # Initialize w to be the mean w
    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable

    # Initialize tensor to save the ws and masks obtained over the optimization
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    mask_out = torch.zeros([num_steps] + list(mask.shape[1:]), dtype=torch.float32, device=device)

    # Create an optimizer object with the
    optimizer = torch.optim.Adam(params = [w_opt] + list(noise_bufs.values()) + [mask],
                                 betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    # Initialize dictionary to save various loss during optimization
    loss_names = ["recon", "disc", "mask", "noise"]
    losses = {loss_name : [0.0] * num_steps for loss_name in loss_names}

    # Optimization iterations
    for step in range(num_steps):
        # Learning rate schedule.
        step_normalized = step / num_steps
        learning_rate = get_lr(step_normalized,
                               lr_rampdown_length,
                               lr_rampup_length,
                               initial_learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

        # Get synth images from opt_w.
        w_noise_scale = get_w_noise_scale(step_normalized,
                                          w_std,
                                          initial_noise_factor,
                                          noise_ramp_length)
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        w_vectors = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(w_vectors, noise_mode='const')

        # Features for synth images.
        synth_features = get_disc_features(D, synth_images)

        # Pixel-wise loss: loss_r = (target_images - synth_images).abs().mean()
        # Recon losses as in Schlegl et al. "Unsupervised anomaly detection with GANs."
        #                               IPMI 2017 https://arxiv.org/pdf/1703.05921.pdf
        # Masked loss as in Mou et al. "Mask-free GAN inverseion"
        #                               ICKR 2023 https://arxiv.org/pdf/2302.12464.pdf
        binary_mask = torch.sigmoid(mask)
        # loss that encourages the mask to be small
        loss_m = binary_mask.mean()
        # Pixel-wise L1 recon loss in un-masked pixels
        loss_r = ((1.0 - binary_mask) * (target_images - synth_images).abs()).mean()
        # Element-wise L1 loss in feature space of Discriminator
        loss_d = (target_features - synth_features).abs().mean()
        # Noise regularization loss from styleGANv2
        loss_n = get_noise_reg_loss(noise_bufs)
        # Total loss
        loss = (1-lambda_d)*loss_r + lambda_d*loss_d + lambda_m*loss_m + lambda_n*loss_n

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step%100==0:
            loss_str = f'step {step+1:>4d}/{num_steps}:'
            loss_str = f'{loss_str} loss_m {loss_m:<4.2f}'
            loss_str = f'{loss_str} loss_r {loss_r:<4.2f}'
            loss_str = f'{loss_str} loss_d {loss_d:<4.2f}'
            loss_str = f'{loss_str} loss_n {loss_n:<4.2f}'
            loss_str = f'{loss_str} loss {loss:<4.2f}'
            logprint(loss_str)

        # Store losses
        losses['recon'][step] = loss_r.item()
        losses['disc'][step] = loss_d.item()
        losses['mask'][step] = loss_m.item()
        losses['noise'][step] = loss_n.item()

        # Save projected W and mask for each optimization step.
        w_out[step] = w_opt.detach()[0]
        mask_out[step] = binary_mask.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out.repeat([1, G.mapping.num_ws, 1]), mask_out, losses

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',     help='Network pickle filename', required=True)
@click.option('--target_path', 'target_path', help='Path of target images', required=True, metavar='FILE')
@click.option('--num_steps',                  help='Num optim. steps', type=int, default=501, show_default=True)
@click.option('--seed',                       help='Random seed', type=int, default=303, show_default=True)
@click.option('--save_video',                 help='Save mp4 video of optim.', type=bool, default=False, show_default=True)
@click.option('--outdir',                     help='Where to save the output images', required=True, metavar='DIR')
def run_projection(
    network_pkl: str,
    target_path: str,
    outdir: str,
    save_video: bool,
    seed: int,
    num_steps: int
):
    """Project given image to the latent space of pretrained network pickle.

    Examples:

    \b
    python projector.py --outdir=out --target=~/mytargetimg.png \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda')
    os.makedirs(outdir, exist_ok=True)

    # Load networks.
    G, D = load_nets(network_pkl, device)

    # Save some samples from the model
    save_samples = False
    if save_samples:
        save_real_and_generated_samples(G, nsamples=48, fname=f"{outdir}", device=device)

    # set here: which images to be reconstructed, and with which settings
    image_indices = [301] # range(0, 1100, 100) # start, end, step
    lams_d_loss = [0.01] # [0.0, 0.01, 0.1] # [0.0, 0.001, 0.01, 0.1]
    lams_mask_loss = [0.5] # [0.0, 0.5, 1.0] # np.round(np.arange(0.2, 1.0, 0.1), 2)
    # magic happens between 0.1 and 1.0 [0.0, 1.0]

    # reconstruct target image(s)
    recon_images = True
    if recon_images:
        settings = itertools.product(image_indices, lams_d_loss, lams_mask_loss)
        for image_idx, lam_d_loss, lam_m_loss in settings:

            # create dir for this image
            outdir_this_image = f"{outdir}image{image_idx}/lambda_d{lam_d_loss}_mask{lam_m_loss}/"
            os.makedirs(outdir_this_image, exist_ok=True)

            # Load target image.
            target = load_image(fname = f"{target_path}image_{image_idx}.png",
                                size = (G.img_resolution, G.img_resolution))

            # Optimize projection.
            start_time = perf_counter()
            projected_w_steps, mask_steps, losses = project(G,
                                                    D,
                                                    target = torch.tensor(target.transpose([2, 0, 1]), device=device), # pylint: disable=line-too-long
                                                    num_steps = num_steps,
                                                    lambda_d = lam_d_loss,
                                                    lambda_m = lam_m_loss,
                                                    device = device)
            print (f'Optimization took: {(perf_counter()-start_time):.1f} s')

            # Render debug output: optional video and projected image and W vector.
            if save_video:
                start_time = perf_counter()
                save_video_showing_optimization(fname = f'{outdir_this_image}proj.mp4',
                                                w_steps = projected_w_steps,
                                                m_steps = mask_steps,
                                                G = G,
                                                losses = losses,
                                                target = target)
                print (f'Saving the video took: {(perf_counter()-start_time):.1f} s')

                # Save final projected frame
                target_image = utils_data.convert_pil_to_arr(np.squeeze(target))
                synth_image = G.synthesis(projected_w_steps[-1].unsqueeze(0), noise_mode='const')
                synth_image = synth_image.permute(0, 2, 3, 1)[0,:,:,0].cpu().numpy()
                mask = mask_steps.permute(0, 2, 3, 1)[-1,:,:,0].cpu().numpy()
                images_to_save = [target_image, synth_image, target_image - synth_image, mask]
                utils_vis.save_images(images = images_to_save,
                                    fname = f'{outdir_this_image}result.png',
                                    nr = 1,
                                    nc = len(images_to_save))

                # Save final W vector.
                np.savez(f'{outdir_this_image}projected_w.npz',
                        w=projected_w_steps[-1].unsqueeze(0).cpu().numpy())

#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
