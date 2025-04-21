import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

def compute_loss(logits, labels):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = input_ids.clone()
            logits, _ = model(input_ids)
            loss = compute_loss(logits, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)



def train_model(model, train_loader, valid_loader, test_loader, model_name, wandb_key=None, epochs=10, lr=3e-4, device='cpu'):
    if wandb_key is not None:
        wandb.login(key=wandb_key)
        wandb.init(project="LM-compare", name=model_name)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        # for batch in train_loader:
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            labels = input_ids.clone()
            logits, _ = model(input_ids)
            loss = compute_loss(logits, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        val_loss = evaluate(model, valid_loader, device)
        test_loss = evaluate(model, test_loader, device)

        if wandb_key is not None:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": val_loss,
                "test_loss": test_loss,
                "train_PPL": torch.exp(torch.tensor(train_loss)).item(),
                "valid_PPL": torch.exp(torch.tensor(val_loss)).item(),
                "test_PPL": torch.exp(torch.tensor(test_loss)).item()
            })


