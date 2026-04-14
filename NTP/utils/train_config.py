from argparse import ArgumentParser
from utils.rule_based_detection import *
from transformers import TrainingArguments
from peft import LoraConfig

parser = ArgumentParser()
parser.add_argument("--language", type=str, required=True, choices=["English", "Japanese"])
parser.add_argument("--pretrainModel", type=str, required=True, choices=["bert", "gpt2", "llama3", "qwen3"])
parser.add_argument("--data_dir", type=str, default=None)
parser.add_argument("--exist_backchannel", type=str, default=None)
parser.add_argument("--interjection", type=str, default=None)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--pretrainModel_dir", type=str, default=None)
parser.add_argument("--training_args", type=str, default=None)
parser.add_argument("--peft_config", type=str, default=None)
parser.add_argument("--ftModel_dir", type=str, default=None)
args = parser.parse_args()

if args.language == "Japanese":
    args.data_dir = "data/Japanese/train_data.csv"
    args.exist_backchannel = ['なんか', 'なんかー', 'なんかね', 'なんかねー', 'あの', 'あのー', 'あのね', 'その', 'そのー', 'でー',  
                              'っていうか', 'ていうか', 'ま', 'まー', 'それで', 'そうなん', 'そうなんだ', 'そうなんだね', 
                              'そうなんだー', 'そうなんだよね', 'そうなんだよねー', 'そうなんだよ', 'そうなんや', 'そうなの',
                              'まだ', 'ねえ', 'は', 'はは', 'ははは', 'はー', 'はあ', 'はあー', 'はっ', 'うーんと',
                              'そうなんですか', 'そうなんですかー', 'そうなんです', 'そうなんですよ', 'そうなんですよー', 
                              'そうなんですね', 'そうなんですねー', 'そうなんですよね', 'そうなんですよねー' ]
    args.interjection = list(set(ACK_BACK['ja'] + AGREE_ACCEPT['ja'] + NEGATIVE_FEEDBACK['ja'] + args.exist_backchannel))
elif args.language == "English":
    args.data_dir = "data/English/train_data.csv"
    args.exist_backchannel = [
        'alright','ehm','ha--','ah','aye','eh', 'nah','erm',"'kay",'yay','a--','okey-dokey', 'mm-mm','uh-uh',
        'well--','phoar','yee-ha','r--', 'rightee-ho','surely','aha','pardon','un--', 'yup','m--','y--','oo','so']
    args.interjection = list(set(ACK_BACK['en'] + AGREE_ACCEPT['en'] + ANSWERS['en'] + NEGATIVE_FEEDBACK['en'] + args.exist_backchannel))


model_paths = {
    "llama3": "lightblue/suzume-llama-3-8B-multilingual", # 32 layers
    "qwen3" : "Qwen/Qwen3-8B-Base", # 36 layers
    "gpt2": {
        "Japanese": "rinna/japanese-gpt2-medium", # 24 layers
        "English": "openai-community/gpt2" # 12 layers
    },
    "bert": {
        "Japanese": "tohoku-nlp/bert-base-japanese-v2", # 12 layers
        "English": "google-bert/bert-base-cased" # 12 layers
    },
}


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
model_root = os.path.join(project_root, "model")

args.output_dir = os.path.join(
    model_root, "checkpoints", f"results_{args.language}_{args.pretrainModel}"
)

args.ftModel_dir = os.path.join(
    model_root, f"fine-tuning_{args.language}_{args.pretrainModel}"
)

os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(args.ftModel_dir, exist_ok=True)

if args.pretrainModel == "llama3" or args.pretrainModel == "qwen3":
    args.pretrainModel_dir = model_paths[args.pretrainModel]
    # Define training arguments
    args.training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        learning_rate=1e-4, 
        num_train_epochs=3,  # Adjust the number of epochs as needed
        per_device_train_batch_size=1,  # Adjust based on your GPU capacity
        save_steps=10_000,
        save_total_limit=2,
        fp16=True
    )
    args.peft_config = LoraConfig(
        r=16,  # the rank of the LoRA matrices
        lora_alpha=32, # the weight
        lora_dropout=0.1, # dropout to add to the LoRA layers
        bias="none", # add bias to the nn.Linear layers?
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"], # the name of the layers to add LoRA
    )
else:
    args.pretrainModel_dir = model_paths[args.pretrainModel][args.language]
    args.training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        learning_rate=3e-5,
        weight_decay=0.01,
        num_train_epochs=3,  # Adjust the number of epochs as needed
        per_device_train_batch_size=64,  # Adjust based on your GPU capacity
        save_steps=10_000,
        save_total_limit=2,
        logging_dir='./logs',
        fp16=True
    )
print(args.pretrainModel_dir)
print(args.peft_config)
