import datasets
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def load_dataset(dataset_name="wikitext", config_name="wikitext-2-raw-v1"):
    dataset = datasets.load_dataset(dataset_name, config_name)
    return dataset

def tokenize_dataset(dataset):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def tokenize_function(data):
        return tokenizer(data["text"])

    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized, tokenizer.vocab_size

def group_dataset(tokenized_dataset, block_size=512):

    def group_texts(data):
        concatenated = {k: sum(data[k], []) for k in data.keys()}
        total_length = len(concatenated["input_ids"])
        total_length = (total_length // block_size) * block_size

        result = {
            k: [concatenated[k][i : i + block_size] for i in range(0, total_length, block_size)]
            for k in concatenated.keys()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    grouped_dataset = tokenized_dataset.map(group_texts, batched=True, batch_size=1000)
    grouped_dataset.set_format(type="torch")
    return grouped_dataset

def get_dataloader(dataset, batch_size=32):
    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset["validation"], batch_size=batch_size)
    test_loader = DataLoader(dataset["test"], batch_size=batch_size)
    return train_loader, val_loader, test_loader

