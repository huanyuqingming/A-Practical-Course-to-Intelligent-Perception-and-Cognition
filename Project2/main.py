import torch
from torchvision import datasets, transforms
import torch.optim as optim
import wandb
import time
import argparse

from model import VAE
from utils import train_epoch, valid_epoch

parser = argparse.ArgumentParser(description='VAE MNIST Generator')
parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
parser.add_argument('--z_dimension', type=int, default=1, help='z dimension for the model')
parser.add_argument('--data_root', type=str, default='./data/', help='root directory for dataset')
args = parser.parse_args()

# Download MNIST Dataset
train_dataset = datasets.MNIST(root=args.data_root, train=True, transform=transforms.ToTensor(), download=True)
valid_dataset = datasets.MNIST(root=args.data_root, train=False, transform=transforms.ToTensor(), download=False)

# MNist Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=False)


# to be finished by you ...
print("Batch size:", args.batch_size)
print("Learning rate:", args.lr)
print("Epochs:", args.epochs)
print("Z-Dimension:", args.z_dimension)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

wandb_name = 'Z=' + str(args.z_dimension) + ' lr=' + str(args.lr) + ' epochs=' + str(args.epochs) + ' batch_size=' + str(args.batch_size)
wandb.init(project="VAE", name=wandb_name,
           config={
               "learning_rate": args.batch_size,
               "epochs": args.epochs,
               "Z-Dimension": args.z_dimension,
               "batch_size": args.batch_size
           })

model = VAE(z_dim=args.z_dimension).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
best_valid = float('inf')
print(f'Training VAE with z_dim={args.z_dimension}')
begin_time = time.time()
for epoch in range(1, args.epochs+1):
    train_loss, train_recon_loss, train_kl_loss = train_epoch(model, train_loader, optimizer, device)
    valid_loss, valid_recon_loss, valid_kl_loss = valid_epoch(model, valid_loader, device)
    if valid_loss < best_valid:
        best_valid = valid_loss
        torch.save(model.state_dict(), f'vae_z{args.z_dimension}_best.pth')

    current_time = time.time()
    elapsed_time = current_time - begin_time
    left_time = elapsed_time / (epoch) * args.epochs - elapsed_time
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    left_time = time.strftime("%H:%M:%S", time.gmtime(left_time))
    
    print(f'Epoch {epoch}/{args.epochs} | {elapsed_time}<{left_time}')
    print(f'Train Loss: {train_loss:.4f} | Train Reconstruction Loss: {train_recon_loss:.4f} | Train KL Loss: {train_kl_loss:.4f}')
    print(f'Valid Loss: {valid_loss:.4f} | Valid Reconstruction Loss: {valid_recon_loss:.4f} | Valid KL Loss: {valid_kl_loss:.4f}')
    wandb.log({"train_loss": train_loss, "train_recon_loss": train_recon_loss, "train_kl_loss": train_kl_loss, 
               "valid_loss": valid_loss, "valid_recon_loss": valid_recon_loss, "valid_kl_loss": valid_kl_loss, 
               "epoch": epoch})

print(f'Best validation loss for z={args.z_dimension}:', best_valid)

wandb.finish()