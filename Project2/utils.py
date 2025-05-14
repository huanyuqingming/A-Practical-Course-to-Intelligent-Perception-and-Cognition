import torch
import torch.nn as nn
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.manifold import TSNE

def loss_fn(x_hat, x, mu, logvar):
    recon_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    train_loss = 0
    train_recon_loss = 0
    train_kl_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        x_hat, mu, logvar, latent = model(data)
        loss, recon_loss, kl_loss = loss_fn(x_hat, data, mu, logvar)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_recon_loss += recon_loss.item()
        train_kl_loss += kl_loss.item()
    return train_loss / len(train_loader.dataset), train_recon_loss / len(train_loader.dataset), train_kl_loss / len(train_loader.dataset)

def valid_epoch(model, valid_loader, device):
    model.eval()
    valid_loss = 0
    valid_recon_loss = 0
    valid_kl_loss = 0
    with torch.no_grad():
        for data, _ in valid_loader:
            data = data.to(device)
            x_hat, mu, logvar, latent = model(data)
            loss, recon_loss, kl_loss = loss_fn(x_hat, data, mu, logvar)
            valid_loss += loss.item()
            valid_recon_loss += recon_loss.item()
            valid_kl_loss += kl_loss.item()
    return valid_loss / len(valid_loader.dataset), valid_recon_loss / len(valid_loader.dataset), valid_kl_loss / len(valid_loader.dataset)


def tensor_to_gif(tensors, save_path, duration=100, loop=0):
    frames = []
    for tensor in tensors:
        tensor = tensor.cpu().detach().float()
        if tensor.dim() == 4:
            tensor = tensor[0]
        
        if tensor.min() < 0:
            tensor = (tensor + 1) / 2
        tensor = tensor * 255
        tensor = tensor.clamp(0, 255).byte()
        
        array = tensor.permute(1, 2, 0).numpy()
        if array.shape[-1] == 1:
            array = array.squeeze(-1)
        frame = Image.fromarray(array, mode='L' if array.ndim == 2 else 'RGB')
        frames.append(frame)
    
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop
    )


def plot_outputs(model, device, z_dim, output_dir='figures', data_loader=None):
    os.makedirs(output_dir, exist_ok=True)
    
    if z_dim == 1:
        zs = torch.linspace(-5, 5, 400).unsqueeze(1).to(device)
        with torch.no_grad():
            imgs = model.decode(zs)
            save_image(imgs, os.path.join(output_dir, 'outputs_1d.png'), nrow=20)

    elif z_dim == 2:
        grid_x = torch.linspace(-5, 5, 20)
        grid_y = torch.linspace(-5, 5, 20)
        mesh_x, mesh_y = torch.meshgrid(grid_x, grid_y, indexing='ij')
        zs = torch.stack([mesh_x.reshape(-1), mesh_y.reshape(-1)], dim=1).to(device)
        with torch.no_grad():
            imgs = model.decode(zs)
            save_image(imgs, os.path.join(output_dir, 'outputs_2d.png'), nrow=20)

    elif z_dim == 3:
        grid_x = torch.linspace(-5, 5, 20)
        grid_y = torch.linspace(-5, 5, 20)
        grid_z = torch.linspace(-5, 5, 20)
        mesh_x, mesh_y, mesh_z = torch.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
        zs = torch.stack([mesh_x.reshape(-1), mesh_y.reshape(-1), mesh_z.reshape(-1)], dim=1).to(device)
        with torch.no_grad():
            imgs = model.decode(zs)
            images = []
            for i in range(20):
                img = imgs[i * 400 : (i + 1) * 400, :, :, :]
                save_image(img, os.path.join(output_dir, f'outputs_3d_{i}.png'), nrow=20)
                img = img.view(20, 20, 1, 28, 28)
                img = img.permute(0, 3, 1, 4, 2).reshape(1, 20 * 28, 20 * 28)
                images.append(img)

        gif_path = os.path.join(output_dir, 'outputs_3d.gif')
        tensor_to_gif(images, gif_path, duration=100, loop=0)

    else:
        if data_loader is None:
            raise ValueError("data_loader must be provided for z_dim > 3")
        model.eval()
        images = []
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(device)
                x_hat, mu, logvar, latent = model(data)
                images.append(x_hat.cpu())
                if len(images) >= 400:
                    break
        images = torch.cat(images, dim=0)
        images = images[:400]
        save_image(images, os.path.join(output_dir, f'outputs_{z_dim}d.png'), nrow=20)
                

def plot_latent_space(model, device, data_loader, z_dim, output_dir='figures'):
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    latents = []
    labels = []
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            _, _, _, latent = model(data)
            latents.append(latent.cpu())
            labels.append(label.cpu())
            
    latents = torch.cat(latents, dim=0)
    labels = torch.cat(labels, dim=0)

    plt.figure(figsize=(8, 6))
    labels_np = labels.numpy()
    cmap = plt.cm.get_cmap('tab10', 10)
    
    if z_dim == 1:
        latents_1d = latents[:, 0].numpy()
        for label_id in range(10):
            mask = labels_np == label_id
            plt.scatter(latents_1d[mask], labels_np[mask],
                       color=cmap(label_id),
                       label=str(label_id),
                       alpha=0.5)
        plt.xlabel('Latent Variable')
        plt.ylabel('Label')
        plt.title('1D Latent Space Visualization')
        
    elif z_dim == 2:
        latents_2d = latents.numpy()
        for label_id in range(10):
            mask = labels_np == label_id
            plt.scatter(latents_2d[mask, 0], latents_2d[mask, 1],
                       color=cmap(label_id),
                       label=str(label_id),
                       alpha=0.5)
        plt.xlabel('Latent Variable 1')
        plt.ylabel('Latent Variable 2')
        plt.title('2D Latent Space Visualization')

    else:
        reducer = TSNE(n_components=2, perplexity=30, n_iter=1000)
        latents_reduced = reducer.fit_transform(latents)

        for label_id in range(10):
            mask = labels == label_id
            plt.scatter(latents_reduced[mask, 0], latents_reduced[mask, 1],
                       color=cmap(label_id),
                       label=str(label_id),
                       alpha=0.5)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title(f'{z_dim}D Latent Space Projection)')
    
    plt.legend(title='Labels', 
              bbox_to_anchor=(1.05, 1),
              loc='upper left',
              borderaxespad=0.)
    plt.tight_layout()
    
    filename = f'latent_space_{z_dim}d.png'
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    plt.close()