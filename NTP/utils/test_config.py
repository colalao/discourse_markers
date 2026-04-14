from argparse import ArgumentParser
from utils.rule_based_detection import *
from transformers import TrainingArguments
from peft import LoraConfig

parser = ArgumentParser()
parser.add_argument("--language", type=str, required=True, choices=["English", "Japanese"])
parser.add_argument("--pretrainModel", type=str, required=True, choices=["bert", "gpt2", "llama3", "qwen3"])
parser.add_argument("--layers", type=list, help="Layer number to extract embeddings from")
parser.add_argument("--pca", action="store_true", help="Enable PCA dimensionality reduction")
parser.add_argument('--n_bootstrap', type=int, default=300, help='Number of bootstrap iterations')
parser.add_argument("--pca_dim", type=int, default=100)
parser.add_argument("--test_type", type=str, required=True, choices=["no_ft_no", "ft_no", "no_ft_one", "ft_one", "no_ft_full", "ft_full"])
parser.add_argument("--data_dir", type=str, default=None)
parser.add_argument("--exist_backchannel", type=str, default=None)
parser.add_argument("--interjection", type=str, default=None)
parser.add_argument("--pretrainModel_dir", type=str, default=None)
parser.add_argument("--loraModel_dir", type=str, default=None)
parser.add_argument("--embedding_path", type=str, default=None)
parser.add_argument("--silscore_path", type=str, default=None)
parser.add_argument("--max_silscore_file", type=str, default=None)
parser.add_argument("--centroids_path", type=str, default=None)
parser.add_argument("--tsne_path", type=str, default=None)
parser.add_argument("--distance_path", type=str, default=None)

args = parser.parse_args()

if args.language == "Japanese":
    args.data_dir = "data/Japanese/Japanese_combined_all_files.csv"
    args.exist_backchannel = ['なんか', 'なんかー', 'なんかね', 'なんかねー', 'あの', 'あのー', 'あのね', 'その', 'そのー', 'でー',  
                              'っていうか', 'ていうか', 'ま', 'まー', 'それで', 'そうなん', 'そうなんだ', 'そうなんだね', 
                              'そうなんだー', 'そうなんだよね', 'そうなんだよねー', 'そうなんだよ', 'そうなんや', 'そうなの',
                              'まだ', 'ねえ', 'は', 'はは', 'ははは', 'はー', 'はあ', 'はあー', 'はっ', 'うーんと',
                              'そうなんですか', 'そうなんですかー', 'そうなんです', 'そうなんですよ', 'そうなんですよー', 
                              'そうなんですね', 'そうなんですねー', 'そうなんですよね', 'そうなんですよねー' ]
    args.interjection = list(set(ACK_BACK['ja'] + AGREE_ACCEPT['ja'] + NEGATIVE_FEEDBACK['ja'] + args.exist_backchannel))
elif args.language == "English":
    args.data_dir = "data/English/English_combined_all_files.csv"
    args.exist_backchannel = [
        'alright','ehm','ha--','ah','aye','eh', 'nah','erm',"'kay",'yay','a--','okey-dokey', 'mm-mm','uh-uh',
        'well--','phoar','yee-ha','r--', 'rightee-ho','surely','aha','pardon','un--', 'yup','m--','y--','oo','so']
    args.interjection = list(set(ACK_BACK['en'] + AGREE_ACCEPT['en'] + ANSWERS['en'] + NEGATIVE_FEEDBACK['en'] + args.exist_backchannel))

print(args.pca_dim)

model_paths = {
    "llama3": {
        "Japanese": ["lightblue/suzume-llama-3-8B-multilingual", [8, 16, 24, -1]],
        "English": ["lightblue/suzume-llama-3-8B-multilingual", [8, 16, 24, -1]],
    },
    "qwen3" : {
        "Japanese": ["Qwen/Qwen3-8B-Base", [9, 18, 27, -1]],
        "English": ["Qwen/Qwen3-8B-Base", [9, 18, 27, -1]],
    },
    "gpt2": {
        "Japanese": ["rinna/japanese-gpt2-medium", [6, 12, 18, -1]],
        "English": ["openai-community/gpt2", [4, 8, -1]]
    },
    "bert": {
        "Japanese": ["tohoku-nlp/bert-base-japanese-v2", [4, 8, -1]],
        "English": ["google-bert/bert-base-cased", [4, 8, -1]]
    },
    "finetuning_model": "model/fine-tuning_"+ args.language + '_' + args.pretrainModel
}

args.layers = model_paths[args.pretrainModel][args.language][1]
print(args.layers)

if args.test_type in ["no_ft_no", "no_ft_one", "no_ft_full"]:
    args.pretrainModel_dir = model_paths[args.pretrainModel][args.language][0]
else:
    args.pretrainModel_dir = model_paths["finetuning_model"]
    if args.pretrainModel == "llama3" or args.pretrainModel == "qwen3":
        args.pretrainModel_dir = model_paths[args.pretrainModel][args.language][0]
        args.loraModel_dir = model_paths["finetuning_model"]
print(args.pretrainModel_dir)
print(args.loraModel_dir)

base_dirs = {
    "embedding_path": "embedding_data",
    "silscore_path": "cluster_results/silscore",
    "centroids_path": "cluster_results/centroids",
    "tsne_path": "visualization/tsne_results",
    "distance_path": "visualization/distance_results"
}

for attr, base in base_dirs.items():
    if args.pca:
        path = f"{base}/{args.pretrainModel}/{args.language}_pca{args.pca_dim}/"
    else:
        path = f"{base}/{args.pretrainModel}/{args.language}/"
    setattr(args, attr, path)

for attr in base_dirs:
    path = getattr(args, attr)
    if not os.path.exists(path):
        os.makedirs(path)

args.embedding_path = { layer: os.path.join(args.embedding_path, f"{args.test_type}_ctx_layer{layer}.pt") for layer in args.layers }
args.max_silscore_path = {layer: os.path.join(args.silscore_path, f"max_silscore_layer{layer}.csv") for layer in args.layers }
args.bootstrap_scores = os.path.join(args.silscore_path, f"bootstrap_silscore_layer-1.npz") 
args.tsne_path = os.path.join(args.tsne_path, f"{args.test_type}_ctx.png")
args.distance_path = os.path.join(args.distance_path, f"{args.test_type}_ctx.png")