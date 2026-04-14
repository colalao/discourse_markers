from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from turngpt.model import TurnGPT
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
        self.model, self.tokenizer = self.load_tokenizer_and_model()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)

    def load_tokenizer_and_model(self):
        model = TurnGPT.load_from_checkpoint(self.args.pretrainModel_dir)
        model.eval()
        tokenizer = model.tokenizer
        return model, tokenizer
    
    def find_token_index(self, utt_len, inputs, interjection_token_list):
        target_left = 0
        check_input_ids = inputs['input_ids'][0]
        check_speaker_ids = inputs['speaker_ids'][0]
        target_right = len(check_input_ids)

        if self.args.test_type == "ft_one" and utt_len > 2 :
            first_speaker_id = check_speaker_ids[0]
            for i, speaker_id in enumerate(check_speaker_ids):
                if speaker_id != first_speaker_id:
                    target_left = i
                    break
            for i in range(len(check_speaker_ids) - 2, -1, -1):
                if check_speaker_ids[i] != first_speaker_id:
                    target_right = i + 1
                    break

        found_indices = []
        for bck, pattern in interjection_token_list:
            sub_len = pattern.size(0)
            for i in range(target_left, target_right - sub_len + 1):
                if torch.equal(check_input_ids[i:i + sub_len], pattern):
                    found_indices.append((bck, [i, i + sub_len - 1]))
        return found_indices
    
    def batch_embedding_pca(self, batch_files=None, data_list=[]): 

        if batch_files is not None:
            for batch_file in batch_files :
                batch_data = torch.load(batch_file)
                data_list.extend(batch_data)

        back_labels = np.stack([item[0] for item in data_list])
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
        data_list = [(back_labels[i], torch.tensor(downsampled_data[i])) for i in range(len(downsampled_labels))]
        
        return data_list
    
    def get_embedding(self, context_text, interjection_token_list):
    
        filtered_data_list = { layer: [] for layer in self.args.layers }
        for utt in tqdm(context_text, desc="Processing dialogues"):
            inputs = self.tokenizer(utt, return_tensors='pt', truncation=True, padding=True, max_length=1024)
            interjection_spans = self.find_token_index(len(utt), inputs, interjection_token_list) 
            with torch.no_grad():
                outputs = self.model(**inputs.to(self.device), output_hidden_states=True)
            hidden_states = outputs.hidden_states

            del inputs, utt, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            for target_bck, target_span in interjection_spans:
                target_bck = re.search(r'<ds>(.*?)</ds>', target_bck).group(1)
                overlap = False
                for _, other_span in interjection_spans:
                    if (target_span[0] >= other_span[0] and target_span[1] < other_span[1]) or (target_span[0] > other_span[0] and target_span[1] <= other_span[1]):
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

        
