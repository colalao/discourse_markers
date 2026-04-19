from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import torch
import romkan
from tqdm import tqdm
import re
import os


class LM():
    def __init__(self, args):
        self.args = args
        self.special_tokens = {'additional_special_tokens': ['/A','/B', '<ds>', '</ds>']}
        self.model, self.tokenizer = self.load_tokenizer_and_model()
        self.model.eval()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)

    def load_llama3_qwen3(self):
        model = AutoModel.from_pretrained(self.args.pretrainModel_dir)
        if self.args.test_type == "no_ft_no" or self.args.test_type == "no_ft_one" or self.args.test_type == "no_ft_full":
            tokenizer = AutoTokenizer.from_pretrained(self.args.pretrainModel_dir)
            if self.args.pretrainModel == "qwen3":
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            tokenizer.add_special_tokens(self.special_tokens)
            model.resize_token_embeddings(len(tokenizer))
        elif self.args.test_type == "ft_no" or self.args.test_type == "ft_one" or self.args.test_type == "ft_full":
            tokenizer = AutoTokenizer.from_pretrained(self.args.loraModel_dir)
            model.resize_token_embeddings(len(tokenizer))
            new_model = PeftModel.from_pretrained(model, self.args.loraModel_dir)
            model = new_model.merge_and_unload()

        return model, tokenizer
    
    def load_gpt2_bert(self):
        tokenizer = AutoTokenizer.from_pretrained(self.args.pretrainModel_dir)
        model = AutoModel.from_pretrained(self.args.pretrainModel_dir)
        if self.args.test_type == "no_ft_no" or self.args.test_type == "no_ft_one":
            tokenizer.add_special_tokens(self.special_tokens)
            if self.args.language == "English" and self.args.pretrainModel == "gpt2":
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
        return model, tokenizer

    def load_tokenizer_and_model(self):
        if self.args.pretrainModel == "llama3" or self.args.pretrainModel == "qwen3":
            model, tokenizer = self.load_llama3_qwen3()
        else:
            model, tokenizer = self.load_gpt2_bert()
        return model, tokenizer
    
    def find_token_index(self, check_input_ids, interjection_token_list):
        target_left = 0
        target_right = len(check_input_ids)

        if self.args.test_type == "no_ft_one" or self.args.test_type == "ft_one":
            speaker_indices = []
            speaker_a_tokens = self.tokenizer.encode('/A', add_special_tokens=False) # /A's token_id
            speaker_b_tokens = self.tokenizer.encode('/B', add_special_tokens=False) # /B's token_id

            for i in range(len(check_input_ids) - len(speaker_b_tokens) + 1):
                if check_input_ids[i:i+len(speaker_b_tokens)].tolist() == speaker_b_tokens:
                    speaker_indices.append(('/B', i))
                elif check_input_ids[i:i+len(speaker_a_tokens)].tolist() == speaker_a_tokens:
                    speaker_indices.append(('/A', i))
            
            if len(speaker_indices) == 6:
                target_left = speaker_indices[2][1] 
                target_right = speaker_indices[3][1] + 1
            elif len(speaker_indices) == 4:
                if speaker_indices[0][0] == '/A':
                    target_left = speaker_indices[2][1]
                    target_right = speaker_indices[3][1] + 1
                elif speaker_indices[0][0] == '/B':
                    target_left = speaker_indices[0][1]
                    target_right = speaker_indices[1][1] + 1
            elif len(speaker_indices) == 2:
                target_left = speaker_indices[0][1]
                target_right = speaker_indices[1][1] + 1

        found_indices = []
        for bck, pattern in interjection_token_list:
            sub_len = pattern.size(0)
            for i in range(target_left, target_right - sub_len + 1):
                if torch.equal(check_input_ids[i:i + sub_len], pattern):
                    found_indices.append((bck, [i, i + sub_len - 1]))
        return found_indices
    
    def batch_embedding_pca(self, data_list=[]): 

        back_labels = np.stack([item[0] for item in data_list])
        if self.args.pretrainModel == "qwen3":
            data = np.stack([item[1].cpu().numpy() for item in data_list]) 
        else:
            data = np.stack([item[1].cpu() for item in data_list]) 
        print(data.shape)

        sample_rate = 1 
        indices = list(range(0, len(data), sample_rate)) 
        downsampled_labels = back_labels[indices] 
        downsampled_data = data[indices] 
        
        if self.args.pca:
            scaler = StandardScaler()
            normalized_embeddings = scaler.fit_transform(downsampled_data)
            pca = PCA(n_components=self.args.pca_dim) 
            downsampled_data = pca.fit_transform(normalized_embeddings) 

        print(downsampled_data.shape)
        if self.args.pretrainModel == "qwen3":
            data_list = [(back_labels[i], torch.tensor(downsampled_data[i], dtype=torch.float32)) for i in range(len(downsampled_labels))]
        else:
            data_list = [(back_labels[i], torch.tensor(downsampled_data[i])) for i in range(len(downsampled_labels))]
        
        return data_list
    
    def get_embedding(self, context_df, interjection_token_list):
        filtered_data_list = { layer: [] for layer in self.args.layers }
        for utt in tqdm(context_df['Dialogue'], desc="Processing dialogues"):
            if len(utt) > 15000:
                utt = utt[:15000]
            if self.args.pretrainModel == "bert":
                inputs = self.tokenizer(utt, return_tensors='pt', truncation=True, padding=True, max_length=512)
            elif self.args.pretrainModel == "gpt2":
                inputs = self.tokenizer(utt, return_tensors='pt', truncation=True, padding=True, max_length=1024)
            else:
                inputs = self.tokenizer(utt, return_tensors='pt', truncation=True, padding=True)
            interjection_spans = self.find_token_index(inputs["input_ids"][0], interjection_token_list) 
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            del inputs, utt, input_ids, attention_mask, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for target_bck, target_span in interjection_spans:
                target_bck = re.search(r'<ds>(.*?)</ds>', target_bck).group(1) # 获取<ds>和</ds>之间的文本
                overlap = False
                for _, other_span in interjection_spans:
                    if (target_span[0] >= other_span[0] and target_span[1] < other_span[1]) or (target_span[0] > other_span[0] and target_span[1] <= other_span[1]):
                        print(f"Overlap detected: {target_bck} {target_span} {other_span}")
                        overlap = True
                        break
                if not overlap:
                    for mid_layer in self.args.layers:
                        mid_embedding = hidden_states[mid_layer][:, target_span[0]:target_span[1]+1, :]
                        filtered_data_list[mid_layer].append((target_bck, mid_embedding[0].mean(dim=0, keepdim=False)))
                
            del hidden_states
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for mid_layer in self.args.layers:
            data_list = self.batch_embedding_pca(data_list=filtered_data_list[mid_layer])
            torch.save(data_list, self.args.embedding_path[mid_layer])
