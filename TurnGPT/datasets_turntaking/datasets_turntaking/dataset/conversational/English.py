from datasets import load_dataset, Dataset
import pandas as pd
import re

def load_English(split = "train"):
    """
    splits = ['train', 'validation', 'test']
    dialogue_id information available, each row corresponds to one utterance, so we just aggrefate the row
    with the same dia_id
    the utterances in vm2 dataset don't contain ',','.', etc
    """
    def add_dataset_name(examples):
        examples["dataset_name"] = "English"
        return examples

    if split == "val":
        split = "validation"
    
    if split == "train":
        dset = pd.read_csv("dataset/English/train_data.csv")
    elif split == "validation":
        dset = pd.read_csv("dataset/English/val_data.csv")
    elif split == "test":
        dset = pd.read_csv("dataset/English/test_data.csv")
    df = pd.DataFrame(dset)
    df = df.groupby("File_ID")["Utterance"].agg(list).reset_index()
    dset = Dataset.from_dict({'dialog': list(df['Utterance'])})
    print(dset)
    dset = dset.map(add_dataset_name)
    return dset

if __name__ == "__main__":
    dset = load_English("train")
    print(dset["dialog"][:1])
    df = pd.DataFrame(dset)
    # 打印前几行数据
    print(df)
    print(len(df))