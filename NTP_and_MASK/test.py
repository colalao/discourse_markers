import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/data1/huggingface"

from utils.test_config import *
from utils.loader import DatasetManager
from utils.get_embedding import LM
from utils.k_means import KM


if __name__ == "__main__":

    LargeModel = LM(args)
    
    dataset = DatasetManager(args, LargeModel.tokenizer)

    if args.test_type == "no_ft_no" or args.test_type == "ft_no":
        context_df = dataset.remove_duplicates(dataset.interjection_df)
    elif args.test_type == "no_ft_one" or args.test_type == "ft_one":
        one_context_df = dataset.concat_one_context()
        context_df = dataset.remove_duplicates(one_context_df)
    elif args.test_type == "no_ft_full" or args.test_type == "ft_full":
        context_df = dataset.concat_full_context()
    print(context_df)

    LargeModel.get_embedding(context_df, dataset.interjection_token_list)

    cluster = KM(args)
 
    for layer in args.layers:
        cluster.k_means(layer, args.embedding_path[layer])

        if layer == -1:
            # avg_silscore, _ = cluster.bootstrap_average_silscore(n_bootstrap=args.n_bootstrap) # bootstrap evaluation
            cluster.tsne_visualization()
            cluster.distance_matrix()

    