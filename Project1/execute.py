import my_dataset
import my_model
import utils
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="RNN", help="Model to train: RNN, LSTM, Transformer, GPT2")
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="Dataset to use")
    parser.add_argument("--config_name", type=str, default="wikitext-2-raw-v1", help="Dataset config")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--wandb_key", type=str, help="WandB API key")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")

    args = parser.parse_args()
    model_name = args.model
    dataset_name = args.dataset_name
    config_name = args.config_name
    wandb_key = args.wandb_key
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    print(f"Model: {model_name}\nDataset: {dataset_name}\nConfig: {config_name}\nEpochs: {epochs}\nBatch Size: {batch_size}\nLearning Rate: {lr}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = my_dataset.load_dataset(dataset_name, config_name)
    tokenized_dataset, vocab_size = my_dataset.tokenize_dataset(dataset)
    grouped_dataset = my_dataset.group_dataset(tokenized_dataset)
    train_loader, valid_loader, test_loader = my_dataset.get_dataloader(grouped_dataset, batch_size)

    if model_name == "RNN":
        model = my_model.RNN(vocab_size)
    elif model_name == "LSTM":
        model = my_model.LSTM(vocab_size)
    elif model_name == "Transformer":
        model = my_model.Transformer(vocab_size)
    elif model_name == "GPT2":
        model = my_model.get_gpt_model(vocab_size)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model.to(device)

    utils.train_model(
        model,
        train_loader,
        valid_loader,
        test_loader,
        model_name=model_name,
        wandb_key=wandb_key,
        epochs=epochs,
        lr=lr,
        device=device
    )


