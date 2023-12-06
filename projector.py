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

def get_D_features(D,
                   img):
    x = None
    for res in D.block_resolutions:
        x, img = getattr(D, f'b{res}')(x, img)
    return x

def save_real_and_generated_samples(G,
                                    nsamples,
                                    fname,
                                    device):
    # save real images
    x_samples = utils_data.load_real_images(nsamples)
    utils_vis.save_images(x_samples, f"{fname}_real.png")
    
    # save generated images
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore
    z_samples = np.random.RandomState(123).randn(nsamples, G.z_dim)
    x_samples = G(torch.from_numpy(z_samples).to(device), None)
    utils_vis.save_images(x_samples.detach().cpu().numpy(), f"{fname}_fake.png")
    
    return 0

def project(
    G,
    D,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    lambda_d_loss              = 0.01,
    verbose                    = False,
    device: torch.device
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore
    D = copy.deepcopy(D).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32) / 127.5 - 1 # scale intensities to -1 to 1
    target_features = get_D_features(D, target_images)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode='const')

        # Features for synth images.
        synth_features = get_D_features(D, synth_images)
        
        # recon losses as in Schlegl et al. "Unsupervised anomaly detection with GANs." IPMI 2017 https://arxiv.org/pdf/1703.05921.pdf
        loss_r = (target_images - synth_images).abs().mean()
        loss_d = (target_features - synth_features).abs().mean()
        dist = (1 - lambda_d_loss) * loss_r + lambda_d_loss * loss_d

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step%100==0: logprint(f'step {step+1:>4d}/{num_steps}: loss_r {loss_r:<4.2f} loss_d {loss_d:<4.2f} loss_rd {dist:<4.2f} loss {float(loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out.repeat([1, G.mapping.num_ws, 1])

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',             help='Network pickle filename', required=True)
@click.option('--target_basepath', 'target_basepath', help='Basepath of target image files to project to', required=True, metavar='FILE')
@click.option('--num-steps',                          help='Number of optimization steps', type=int, default=501, show_default=True)
@click.option('--seed',                               help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video',                         help='Save an mp4 video of optimization progress', type=bool, default=False, show_default=True)
@click.option('--outdir',                             help='Where to save the output images', required=True, metavar='DIR')
@click.option('--lambda_d_loss',                      help='Regularization weight', type=float, default=0.01, show_default=True)
def run_projection(
    network_pkl: str,
    target_basepath: str,
    outdir: str,
    save_video: bool,
    seed: int,
    num_steps: int,
    lambda_d_loss: float
):
    """Project given image to the latent space of pretrained network pickle.

    Examples:

    \b
    python projector.py --outdir=out --target=~/mytargetimg.png \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        data = legacy.load_network_pkl(fp)
    G = data['G_ema'].requires_grad_(False).to(device) # type: ignore
    D = data['D'].requires_grad_(False).to(device) # type: ignore

    # Save some samples from the model
    os.makedirs(outdir, exist_ok=True)
    save_samples = False
    if save_samples:
        save_real_and_generated_samples(G, nsamples=48, fname=f"{outdir}/", device=device)

    # reconstruct target image
    recon_images = True
    if recon_images:

        image_indices = np.arange(0, 1000, 100) # start, end, step
        lams_d_loss = [0.0, 0.001, 0.01, 0.1]
        # basepath = "/data/vision/polina/users/nkarani/projects/anomaly/ideas/stylegan/datasets/fets_from_hdf5/"
        basepath = target_basepath

        for image_idx in image_indices:

            for lam_d_loss in lams_d_loss:
                # create dir for this image
                outdir_this_image = f"{outdir}/image{image_idx}/lambda_d{lam_d_loss}/"
                os.makedirs(outdir_this_image, exist_ok=True)

                # Load target image.
                target_fname = f"{basepath}/image_{image_idx}.png"
                target_pil = PIL.Image.open(target_fname).convert('L') # .convert('RGB') for rgb images
                w, h = target_pil.size
                s = min(w, h)
                target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
                target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.Resampling.LANCZOS)
                target_uint8 = np.expand_dims(np.array(target_pil, dtype=np.uint8), axis=-1) # expand_dims needed for grayscale images

                # Optimize projection.
                start_time = perf_counter()
                projected_w_steps = project(
                    G,
                    D,
                    target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
                    num_steps=num_steps,
                    lambda_d_loss=lam_d_loss, 
                    device=device,
                    verbose=True
                )
                print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

                # Render debug output: optional video and projected image and W vector.
                if save_video:
                    video = imageio.get_writer(f'{outdir_this_image}/proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
                    print (f'Saving optimization progress video "{outdir_this_image}/proj.mp4"')
                    for projected_w in projected_w_steps:
                        synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
                        synth_image = (synth_image + 1) * (255/2)
                        synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
                        video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
                    video.close()

                # Save final projected frame
                target_image = utils_data.convert_pil_to_arr(target_pil)
                synth_image = G.synthesis(projected_w_steps[-1].unsqueeze(0), noise_mode='const')
                synth_image = synth_image.permute(0, 2, 3, 1)[0,:,:,0].cpu().numpy()
                utils_vis.save_images(images = [target_image, synth_image, target_image - synth_image],
                                      fname = f'{outdir_this_image}/result.png', nr = 1, nc = 3)

                # Save final W vector.
                np.savez(f'{outdir_this_image}/projected_w.npz', w=projected_w_steps[-1].unsqueeze(0).cpu().numpy())
                

#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
