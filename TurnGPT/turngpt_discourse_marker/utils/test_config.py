from argparse import ArgumentParser
from turngpt_discourse_marker.utils.rule_based_detection import *
from transformers import TrainingArguments
from peft import LoraConfig

parser = ArgumentParser()
parser.add_argument("--language", type=str, required=True, choices=["English", "Japanese"])
parser.add_argument("--pca", action="store_true", help="Enable PCA dimensionality reduction")
parser.add_argument("--pca_dim", type=int, default=100)
parser.add_argument('--n_bootstrap', type=int, default=300, help='Number of bootstrap iterations')
parser.add_argument("--test_type", type=str, required=True, choices=["ft_no", "ft_one", "ft_full"])
parser.add_argument("--data_dir", type=str, default=None)
parser.add_argument("--exist_backchannel", type=str, default=None)
parser.add_argument("--interjection", type=str, default=None)
parser.add_argument("--pretrainModel_dir", type=str, default=None)
parser.add_argument("--layers", type=list, help="Layer number to extract embeddings from")
parser.add_argument("--embedding_path", type=str, default=None)
parser.add_argument("--silscore_path", type=str, default=None)
parser.add_argument("--max_silscore_file", type=str, default=None)
parser.add_argument("--centroids_path", type=str, default=None)
parser.add_argument("--tsne_path", type=str, default=None)
parser.add_argument("--distance_path", type=str, default=None)
args = parser.parse_args()

if args.language == "Japanese":
    args.data_dir = "dataset/Japanese/Japanese_combined_all_files.csv"
    args.exist_backchannel = ['なんか', 'なんかー', 'なんかね', 'なんかねー', 'あの', 'あのー', 'あのね', 'その', 'そのー', 'でー',  
                              'っていうか', 'ていうか', 'ま', 'まー', 'それで', 'そうなん', 'そうなんだ', 'そうなんだね', 
                              'そうなんだー', 'そうなんだよね', 'そうなんだよねー', 'そうなんだよ', 'そうなんや', 'そうなの',
                              'まだ', 'ねえ', 'は', 'はは', 'ははは', 'はー', 'はあ', 'はあー', 'はっ', 'うーんと',
                              'そうなんですか', 'そうなんですかー', 'そうなんです', 'そうなんですよ', 'そうなんですよー', 
                              'そうなんですね', 'そうなんですねー', 'そうなんですよね', 'そうなんですよねー' ]
    args.interjection = list(set(ACK_BACK['ja'] + AGREE_ACCEPT['ja'] + NEGATIVE_FEEDBACK['ja'] + args.exist_backchannel))
elif args.language == "English":
    args.data_dir = "dataset/English/English_combined_all_files.csv"
    args.exist_backchannel = [
        'alright','ehm','ha--','ah','aye','eh', 'nah','erm',"'kay",'yay','a--','okey-dokey', 'mm-mm','uh-uh',
        'well--','phoar','yee-ha','r--', 'rightee-ho','surely','aha','pardon','un--', 'yup','m--','y--','oo','so']
    args.interjection = list(set(ACK_BACK['en'] + AGREE_ACCEPT['en'] + ANSWERS['en'] + NEGATIVE_FEEDBACK['en'] + args.exist_backchannel))

print(args.pca_dim)

model_paths = {
    "Japanese": ["finetune_model/Japanese_20250807-195522/epoch=5_val_loss=1.8754.ckpt", [6, 12, 18, -1]],
    "English": ["finetune_model/English_20250807-203506/epoch=46_val_loss=2.6777.ckpt", [4, 8, -1]]
}

args.layers = model_paths[args.language][1]
args.pretrainModel_dir = model_paths[args.language][0]
print(args.pretrainModel_dir)


base_dirs = {
    "embedding_path": "turngpt_discourse_marker/embedding_data",
    "silscore_path": "turngpt_discourse_marker/cluster_results/silscore",
    "centroids_path": "turngpt_discourse_marker/cluster_results/centroids",
    "tsne_path": "turngpt_discourse_marker/visualization/tsne_results",
    "distance_path": "turngpt_discourse_marker/visualization/distance_results"
}

for attr, base in base_dirs.items():
    if args.pca:
        if attr == "embedding_path":
            path = f"{base}/turngpt/{args.language}_pca{args.pca_dim}/"
        else:
            path = f"{base}/{args.language}_pca{args.pca_dim}/"
    else:
        if attr == "embedding_path":
            path = f"{base}/turngpt/{args.language}/"
        else:            
            path = f"{base}/{args.language}/"

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
