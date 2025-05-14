import torch
from torchvision import datasets, transforms
import os
import argparse

from model import VAE
from utils import plot_outputs, plot_latent_space

parser = argparse.ArgumentParser(description='VAE MNIST Generator')
parser.add_argument('--z_dimension', type=int, default=1, help='z dimension for the model')
parser.add_argument('--data_root', type=str, default='./data/', help='root directory for dataset')
parser.add_argument('--output_dir', type=str, default='figures', help='directory to save output images')
args = parser.parse_args()

print("Z-Dimension:", args.z_dimension)
print("Data root:", args.data_root)
print("Output directory:", args.output_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = datasets.MNIST(root=args.data_root, train=False, transform=transforms.ToTensor(), download=False)
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=256, shuffle=False)

model_path = f"./vae_z{args.z_dimension}_best.pth"
assert os.path.exists(model_path), f"Model file {model_path} does not exist."

model = VAE(z_dim=args.z_dimension).to(device)
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()
plot_outputs(model, device, z_dim=args.z_dimension, data_loader=data_loader, output_dir=args.output_dir)
plot_latent_space(model, device, data_loader, z_dim=args.z_dimension, output_dir=args.output_dir)