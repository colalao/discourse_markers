import pandas as pd
import torch
import re
from datasets import Dataset


class DatasetManager(object):

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.interjection_token_list = []
        if args.language == "English":
            self.tokenizer_English_interjection()
        elif args.language == "Japanese":
            self.tokenizer_Japanese_interjection()
        print(len(self.interjection_token_list))
        self.original_df, self.interjection_df = self.read_df()
    
    def tokenizer_English_interjection(self):
        for bck in ['<ds>' + i + '</ds>' for i in self.args.interjection]:
            ## here every backchannel corresponds
            token = self.tokenizer(bck, return_tensors="pt", truncation=True)
            self.interjection_token_list.append((bck, token.input_ids[0]))
            print(self.interjection_token_list[-1])

    def tokenizer_Japanese_interjection(self):
        for bck in ['<ds>' + i + '</ds>' for i in self.args.interjection]:
            ## here every backchannel corresponds
            token = self.tokenizer(bck, return_tensors="pt", truncation=True)
            self.interjection_token_list.append((bck, token.input_ids[0][:-1]))
            print(self.interjection_token_list[-1])

    def Ja_processing(self, combined_df):
        merged_data = []
        current_speaker = None
        current_utterance = []
        current_file_id = None
        for index, row in combined_df.iterrows():
            speaker = row["Speaker"]
            utterance = row["Dialogue"]
            file_id = row["File_ID"]
            if speaker == current_speaker and file_id == current_file_id:
                current_utterance.append(utterance)
            else:
                if current_speaker is not None:
                    merged_data.append([current_speaker, " ".join(current_utterance), current_file_id])
                current_speaker = speaker
                current_utterance = [utterance]
                current_file_id = file_id
        if current_speaker is not None:
            merged_data.append([current_speaker, " ".join(current_utterance), current_file_id])

        original_df = pd.DataFrame(merged_data, columns=["Speaker", "Dialogue", "File_ID"])
        return original_df

    def contains_backchannel(self, text):
        text_ids = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        text_ids = text_ids["input_ids"][0]
        for _, pattern in self.interjection_token_list:
            sub_len = pattern.size(0)
            for i in range(len(text_ids) - sub_len + 1):
                if torch.equal(text_ids[i:i + sub_len], pattern):
                    return True
        return False

    def read_df(self):
        combined_df = pd.read_csv(self.args.data_dir)
        if self.args.language == "English":
            combined_df = combined_df.rename(columns={'Utterance': 'Dialogue'})
            original_df = combined_df
        elif self.args.language == "Japanese":
            combined_df = combined_df
            original_df = self.Ja_processing(combined_df)

        mask = original_df['Dialogue'].apply(lambda x: self.contains_backchannel(x)) 
        interjection_df = original_df[mask]

        return original_df, interjection_df
    
    def remove_duplicates(self, text):
        unique_text = [list(item) for item in set(tuple(sublist) for sublist in text)]
        return unique_text

    def concat_no_context(self):
        no_context_text = []
        for utt in self.interjection_df['Dialogue']:
            no_context_text.append([utt])
        print(len(no_context_text))
        return no_context_text
    
    def concat_one_context(self):
        # 将小词和其前后语句拼在一起
        one_context_text = []
        for i in range(len(self.interjection_df)):
            original_index = self.interjection_df.index[i]
            current_dialogue = [self.original_df.at[original_index, 'Dialogue']]
            if original_index > 0 and self.original_df.at[original_index, 'File_ID'] == self.original_df.at[original_index - 1, 'File_ID']:
                current_dialogue =  [self.original_df.at[original_index - 1, 'Dialogue']] + current_dialogue
            if original_index < len(self.original_df) - 1 and self.original_df.at[original_index, 'File_ID'] == self.original_df.at[original_index + 1, 'File_ID']:
                current_dialogue = current_dialogue + [self.original_df.at[original_index + 1, 'Dialogue']]
            one_context_text.append(current_dialogue)
        return one_context_text
    




        
        