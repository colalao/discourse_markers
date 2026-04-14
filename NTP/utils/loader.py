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
            if self.args.pretrainModel == "bert":
                self.interjection_token_list.append((bck, token.input_ids[0][1:-1]))
            else: 
                self.interjection_token_list.append((bck, token.input_ids[0]))
            print(self.interjection_token_list[-1])

    def tokenizer_Japanese_interjection(self):
        for bck in ['<ds>' + i + '</ds>' for i in self.args.interjection]:
            ## here every backchannel corresponds
            token = self.tokenizer(bck, return_tensors="pt", truncation=True)
            if self.args.pretrainModel == "gpt2":
                self.interjection_token_list.append((bck, token.input_ids[0][:-1]))
            elif self.args.pretrainModel == "bert": 
                self.interjection_token_list.append((bck, token.input_ids[0][1:-1]))
            else:
                self.interjection_token_list.append((bck, token.input_ids[0]))
            print(self.interjection_token_list[-1])
    

    def En_processing(self, combined_df):
        merged_data = []
        current_speaker = None
        current_utterance = []
        current_file_id = None
        for _, row in combined_df.iterrows():
            speaker = row["Speaker"]
            utterance = row["Utterance"]
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
        if self.args.language == "Japanese":
            original_df = combined_df
        elif self.args.language == "English":
            original_df = self.En_processing(combined_df)

        mask = original_df['Dialogue'].apply(lambda x: self.contains_backchannel(x)) 
        interjection_df = original_df[mask]

        return original_df, interjection_df
    
    def remove_duplicates(self, df):
        df = df.drop_duplicates(subset=['Dialogue'], keep='first')
        return df
    
    def concat_one_context(self):
        modified_dialogues = []
        for i in range(len(self.interjection_df)):
            original_index = self.interjection_df.index[i]
            speaker1 = speaker2 = speaker3 = '/B'
            current_dialogue = speaker2 + ' ' + self.original_df.at[original_index, 'Dialogue'] + ' ' + speaker2 + ' '
            if original_index > 0 and self.original_df.at[original_index, 'File_ID'] == self.original_df.at[original_index - 1, 'File_ID']:
                if self.original_df.at[original_index, 'Speaker'] != self.original_df.at[original_index - 1, 'Speaker']:
                    speaker1 = '/A'
                current_dialogue = speaker1 + ' ' + self.original_df.at[original_index - 1, 'Dialogue'] + ' ' + speaker1 + ' ' + current_dialogue
            if original_index < len(self.original_df) - 1 and self.original_df.at[original_index, 'File_ID'] == self.original_df.at[original_index + 1, 'File_ID']:
                if self.original_df.at[original_index, 'Speaker'] != self.original_df.at[original_index + 1, 'Speaker']:
                    speaker3 = '/A'
                current_dialogue = current_dialogue + speaker3 + ' ' + self.original_df.at[original_index + 1, 'Dialogue'] + ' ' + speaker3 
            modified_dialogues.append(current_dialogue)
        one_context_df = pd.DataFrame({'Dialogue': modified_dialogues}) # （前一句+小词+后一句）
        return one_context_df
    
    def concat_full_context(self):
        
        speakers_dict = self.original_df.groupby('File_ID')['Speaker'].unique().to_dict()
        
        file_id = self.original_df.at[0, 'File_ID']
        current_dialogue = ''
        modified_dialogues = []
        speaker = {speakers_dict[file_id][0]: '/A', speakers_dict[file_id][1]: '/B', '話者不明': '/C'}
        for i in range(len(self.original_df)):
            original_index = self.original_df.index[i]
            if self.original_df.at[original_index, 'File_ID'] == file_id:
                current_dialogue = current_dialogue + ' ' + speaker[self.original_df.at[original_index, 'Speaker']] + ' ' + self.original_df.at[original_index, 'Dialogue'] + ' ' + speaker[self.original_df.at[original_index, 'Speaker']]
            else:
                file_id = self.original_df.at[original_index, 'File_ID']
                modified_dialogues.append(current_dialogue)
                speaker = {speakers_dict[file_id][0]: '/A', speakers_dict[file_id][1]: '/B', '話者不明': '/C'}
                current_dialogue = speaker[self.original_df.at[original_index, 'Speaker']] + ' ' + self.original_df.at[original_index, 'Dialogue'] + ' ' + speaker[self.original_df.at[original_index, 'Speaker']]
            if i == len(self.original_df)-1:
                modified_dialogues.append(current_dialogue)
        full_context_df = pd.DataFrame({'Dialogue': modified_dialogues})
        return full_context_df
    
    def tokenize_function(self, examples):
        return self.tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)
    
    def build_trainData(self):
        one_context_df = self.concat_one_context()
        print(one_context_df)
        # Example text data, you can replace with your dataset
        texts = list(one_context_df['Dialogue'])
        dataset = Dataset.from_dict({"text": texts})
        tokenized_dataset = dataset.map(self.tokenize_function, batched=True) # {'text':' ', 'input_ids':[ ], 'attention_mask':[ ]}
        return tokenized_dataset




        
        