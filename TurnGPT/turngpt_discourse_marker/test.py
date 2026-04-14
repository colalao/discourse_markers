import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "/data1/huggingface"

from turngpt_discourse_marker.utils.test_config import *
from turngpt_discourse_marker.utils.loader import DatasetManager
from turngpt_discourse_marker.utils.get_embedding import LM
from turngpt_discourse_marker.utils.k_means import KM


if __name__ == "__main__":
    
    LargeModel = LM(args)
    
    dataset = DatasetManager(args, LargeModel.tokenizer)

    if args.test_type == "ft_no":
        no_context_df = dataset.concat_no_context()
        context_text = dataset.remove_duplicates(no_context_df)
    elif args.test_type == "ft_one":
        one_context_text = dataset.concat_one_context()
        context_text = dataset.remove_duplicates(one_context_text)

    LargeModel.get_embedding(context_text, dataset.interjection_token_list)
    
    cluster = KM(args)

    for layer in args.layers:
        cluster.k_means(layer, args.embedding_path[layer])

        if layer == -1:
            # avg_silscore, _ = cluster.bootstrap_average_silscore(n_bootstrap=args.n_bootstrap)
            cluster.tsne_visualization()
            cluster.distance_matrix()
    
    