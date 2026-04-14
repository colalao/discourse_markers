import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/data1/huggingface"

import torch
from utils.train_config import *
from utils.loader import DatasetManager
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling, Trainer
from peft import get_peft_model

class FocalDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, interjection_tokens_list, mlm_probability=1.0):
        super().__init__(tokenizer, mlm=True, mlm_probability=mlm_probability)
        self.tokenizer = tokenizer
        self.interjection_tokens_list = interjection_tokens_list  # 小词的 token ids

    def torch_mask_tokens(self, inputs, special_tokens_mask=None):
        """
        inputs: (batch_size, seq_length)
        special_tokens_mask: List[List[int]]
        """
        labels = inputs.clone()
        mask = torch.full(labels.shape, False)

        for i, input_ids in enumerate(inputs):  
            for _, interjection_token in self.interjection_tokens_list:
                interjection_token = interjection_token.to(inputs.device) 
                length = len(interjection_token)

                for j in range(len(input_ids) - length + 1):
                    if torch.equal(input_ids[j:j+length], interjection_token):
                        rand = torch.rand(1).item()
                        if rand < 0.8:
                            # 80%: Mask whole group
                            inputs[i, j:j+length] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
                        elif rand < 0.9:
                            # 10%: Replace with random tokens (same length)
                            random_words = torch.randint(len(self.tokenizer), (length,), dtype=torch.long, device=inputs.device)
                            inputs[i, j:j+length] = random_words
                        mask[i, j:j+length] = True

        # If you don't mask any small words, randomly select some token masks to prevent the training from crashing
        if not mask.any():
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            if special_tokens_mask is None:
                special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
                ]
                special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool, device=inputs.device)
            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100

            # Apply standard masking strategy
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8, device=inputs.device)).bool() & masked_indices
            inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5, device=inputs.device)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long, device=inputs.device)
            inputs[indices_random] = random_words[indices_random]
        else:
            labels[~mask] = -100  # Only compute loss for masked interjection tokens
        
        
        return inputs, labels

def LM():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrainModel_dir)
    model = AutoModelForCausalLM.from_pretrained(args.pretrainModel_dir)

    if args.language == "Japanese":
        new_tokens = [word for word in args.interjection if len(tokenizer.tokenize(word)) > 1]
    else: 
        new_tokens = [word for word in args.interjection if len(word.split()) == 1 and word not in tokenizer.get_vocab() ]
    tokenizer.add_tokens(new_tokens)
    print(f"New tokens to add: {new_tokens}, {len(new_tokens)}")

    if (args.language == "English" and args.pretrainModel == "gpt2") or args.pretrainModel == "qwen3":
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.add_special_tokens({'additional_special_tokens': ['/A','/B', '<ds>', '</ds>']})
    model.resize_token_embeddings(len(tokenizer))

    if args.pretrainModel == "llama3" or args.pretrainModel == "qwen3":
        model = get_peft_model(model, args.peft_config)

    model.to(device)
    return model, tokenizer 


if __name__ == "__main__":

    model, tokenizer = LM()

    dataset = DatasetManager(args, tokenizer)
    tokenized_dataset = dataset.build_trainData()

    # Data collator to handle padding
    if args.pretrainModel == "bert":
        data_collator = FocalDataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            interjection_tokens_list=dataset.interjection_token_list,
            mlm_probability=0.15 
        )
    else:
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, 
            mlm=False  # mlm=False means this is not a masked language modeling task
        )

    trainer = Trainer(
        model=model,
        args=args.training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
    )

    if  args.language == "Japanese" and args.pretrainModel == "bert":
        for param in model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()

    trainer.train()
    model.save_pretrained(args.ftModel_dir)
    tokenizer.save_pretrained(args.ftModel_dir)