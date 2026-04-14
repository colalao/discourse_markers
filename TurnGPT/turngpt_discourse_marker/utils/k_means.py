import cupy as cp
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans as sk_KMeans
from sklearn.metrics import silhouette_score as sk_silhouette_score
from cuml import KMeans as cu_KMeans
from cuml.metrics.cluster import silhouette_score as cu_silhouette_score
from tqdm import tqdm
import torch
import csv
import pandas as pd
import seaborn as sns
import random
import romkan
import matplotlib.pyplot as plt


class KM():
    def __init__(self, args):
        self.args = args
        self.main_words_dic, self.main_words = self.merge_word()
        self.max_silscores = {}
        self.max_silscores_k = {}
        self.max_silscores_centroids = {}
        self.all_embeddings = []
        self.back_labels_roma = []
        self.word_to_embeddings = {}

    def merge_word(self):
        if self.args.language == 'Japanese':
            main_words_dic = {
                'うん':'うん', 'うんうん':'うん', 'うんうんうん':'うん', 'ううん':'うん', 'うーん':'うん', 'うんー':'うん',
                'そう':'そう', 'そうそう':'そう', 'そうそうそう':'そう', 'そうー':'そう', 'そーう':'そう', 'ね':'ね', 'ねー':'ね', 
                'はい':'はい', 'はいー':'はい', 'はーい':'はい', 'あ':'あ', 'ああ':'あ', 'あああ':'あ', 'あー':'あ', 'あっ':'あ',
                'なんか':'なんか', 'なんかー':'なんか', 'なんかね':'なんか','なんかねー':'なんか', 'あの':'あの', 'あのー':'あの', 'あのね':'あの',
                'ま':'ま', 'まー':'ま', 'まあ':'ま', 'まあー':'ま', 'え':'え', 'ええ':'え', 'えええ':'え', 'えー':'え', 'えっ':'え', 
                'ん':'ん', 'んー':'ん', 'へー':'へー', 'その':'その', 'そのー':'その', 
                'そうです':'そうです', 'そうですね':'そうです', 'そうですねー':'そうです', 'そうですよ':'そうです', 'そうですよね':'そうです', 'そうですよねー':'そうです', 'そーうですね':'そうです', 'そうっすね':'そうです', 
                'いや':'いや', 'いやいや':'いや', 'いやいやいや':'いや', 'いやー':'いや', 'あ、そう':'あ、そう', 'でー':'でー',
                'なるほど':'なるほど', 'なるほどー':'なるほど', 'なるほどね':'なるほど', 'なるほどねー':'なるほど', 'ふーん':'ふーん',
                'そうか':'そうか', 'そうかー':'そうか', 'そっか':'そうか', 'そっかー':'そうか', 'ちょっと':'ちょっと',
                'そうだ':'そうだ', 'そうだね':'そうだ', 'そうだねー':'そうだ', 'そうだよ':'そうだ', 'そうだな':'そうだ', 'そうだよね':'そうだ', 'そうだよねー':'そうだ',
                'そうなんですか':'そうなんですか', 'そうなんですかー':'そうなんですか', 'そうなんです':'そうなんです', 'そうなんですよ':'そうなんです',
                'そうなんですよー':'そうなんです', 'そうなんですね':'そうなんです', 'そうなんですねー':'そうなんです', 'そうなんですよね':'そうなんです', 'そうなんですよねー':'そうなんです',
                'そうなん':'そうなん', 'そうなんだ':'そうなん', 'そうなんだね':'そうなん', 'そうなんだー':'そうなん', 'そうなんだよね':'そうなん', 'そうなんだよねー':'そうなん', 'そうなんだよ':'そうなん', 'そうなんや':'そうなん',
                'えと':'えと', 'えーと':'えと', 'えーとー':'えと', 'ええと':'えと', 'えーっと':'えと', 'えーとね':'えと', 'えっと':'えと', 'ええっと':'えと',
                'ていうか':'ていうか', 'っていうか':'ていうか', 'そうですか':'そうですか', 'そうですかー':'そうですか', '確かに':'確かに', 'へえ':'へえ', 'へえー':'へえ', 
                'です':'です', 'ですね':'です', 'ですねー':'です', 'ですよね':'です', 'ですよねー':'です', 'うわ':'うわ', 'うわー':'うわ',
                'うそ':'うそ', 'うそー':'うそ', 'いいえ':'いいえ',  '大変です':'大変です', '大変ですね':'大変です', '大変ですよ':'大変です', '大変ですよね':'大変です',
                'たしかに':'たしかに', 'ふうん':'ふうん', 'ふうーん':'ふうん', 'たいへん':'たいへん', 'その通り':'その通り', 'それで':'それで', 'そうなの':'そうなの',
                'まじ':'まじ', 'ねえ':'ねえ', 'は':'は', 'はは':'は', 'ははは':'は', 'はー':'は', 'はあ':'は', 'はあー':'は', 'はっ':'は', 'うーんと':'うーんと', 'まだ':'まだ', 
            }
            main_words = ['うん', 'あ', 'はい', 'え', 'そう', 'ま', 'なんか', 'あの', 'ん', 'そうです', 'は', 'ね', 'いや', 'へー', 'そうか',
                          #'その', 'えと', 'そうだ', 'そうなん', 'なるほど', 'ふーん', 'です', '確かに', 'ちょっと', 'そうですか', 'そうなんです', 'そうなんですか', 'でー', 'へえ', 'それで']
                          #'そうなの', 'ねえ', 'あ、そう', 'まだ', 'ていうか', 'うわ', 'うそ', '大変です', 'まじ', 'うーんと', 'たしかに', 'いいえ', 'ふうん', 'たいへん', 'その通り'
                        ]
        elif self.args.language == 'English':
            main_words_dic = {}
            main_words = [
                'uh', 'yeah', 'uh-huh', 'well', 'right', 'oh', 'um', 'okay', 'no', 'yes', 'so', 'oh yeah', 'huh', 'mmhmm', 'of course', 
                #'sure', 'eh', 'ehm', 'you know', 'really', 'oh really', 'wow', 'huh-uh', 'exactly', 'all right', 'i know', 'i see', 'it is', 'oh no', 'aye'
            ]
        return main_words_dic, main_words
    
    
    def tsne_visualization(self):

        self.all_embeddings = np.asarray(np.vstack(self.all_embeddings), dtype=np.float32)
        self.back_labels_roma = np.array(self.back_labels_roma)

        tsne = TSNE(n_components=2, random_state=42, perplexity=10)
        tsne_results = tsne.fit_transform(np.vstack(self.all_embeddings))
        
        random.seed(42)
        unique_labels = sorted(list(set(self.back_labels_roma)))
        colors = plt.cm.get_cmap('tab20', len(unique_labels))
        color_dict = {label: colors(i) for i, label in enumerate(unique_labels)}

        plt.figure(figsize=(10, 8))
        for label in unique_labels:
            idx = [i for i, lbl in enumerate(self.back_labels_roma) if lbl == label]
            plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], color=color_dict[label], label=label, marker='o')

        plt.title(f"t-SNE Visualization of Clustering for '{self.args.test_type}'")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.legend()
        plt.savefig(self.args.tsne_path)
        plt.show()
    

    def k_means_silscore(self, k):
        """
        For a given value of k, calculate the profile score for each word for that k and update the optimal value
        k: int, number of clusters
        """

        # output_file = self.args.silscore_path + "/k_" + str(k) + ".txt"

        sil_scores = [self.args.test_type + "_ctx"]
        for main_word in self.main_words:
            unique_embedding_data = self.word_to_embeddings.get(main_word)
            if len(unique_embedding_data) < 2:
                continue 
            
            if k == 2:
                self.max_silscores[main_word] = -1
                self.max_silscores_k[main_word] = -1
                self.all_embeddings.extend(unique_embedding_data)
                self.back_labels_roma.extend([romkan.to_roma(main_word).replace('、', ',').replace('。', '.')] * len(unique_embedding_data))

            kmeans = sk_KMeans(n_clusters=k, random_state=42)
            kmeans.fit(unique_embedding_data)
            labels = kmeans.labels_

            # silscore
            sil_score = sk_silhouette_score(unique_embedding_data, labels)
            if sil_score > self.max_silscores[main_word]:
                self.max_silscores[main_word] = sil_score
                self.max_silscores_k[main_word] = k
                self.max_silscores_centroids[main_word] = torch.tensor(kmeans.cluster_centers_)

            sil_scores.append(f"Silhouette Score for '{main_word}': {sil_score:.4f}")

        # with open(output_file, "a", encoding="utf-8") as f:
        #     f.write("\n".join(sil_scores) + "\n\n")
        

    def k_means(self, layer, data_path):

        embedding_datas = torch.load(data_path, weights_only=False)

        np.random.seed(42)

        cpu_embeddings = {}
        for word, emb in embedding_datas:
            if self.args.language == 'Japanese':
                main_word = self.main_words_dic[word]
            else:
                main_word = word
            if main_word not in self.word_to_embeddings:
                self.word_to_embeddings[main_word] = []
            cpu_embeddings.setdefault(main_word, []).append(emb.cpu().numpy())
  
        for word, emb_list in cpu_embeddings.items():
            arr = np.array(emb_list)
            unique_arr, _ = np.unique(arr, axis=0, return_index=True)
            self.word_to_embeddings[word] = unique_arr

        print(f"Finished loading and deduplicating embeddings for layer {layer}. Starting k-means clustering and silhouette score calculation...")
        for k in tqdm(range(2,16)):
            self.k_means_silscore(k)
        
        if layer == -1:
            torch.save(self.max_silscores_centroids, self.args.centroids_path + str(self.args.test_type + f"_layer{layer}.pt"))
        with open(self.args.max_silscore_path[layer], "a", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([self.args.test_type + "_ctx"])
            for word, silscore in self.max_silscores.items():
                writer.writerow([word, silscore, self.max_silscores_k[word]])
        

    def bootstrap_average_silscore(self, n_bootstrap=1000, random_state=42):
        """
        Bootstrap: each iteration samples all words separately and computes the average profile score.
        Return:
            avg_scores: list of float, Average profile score per iteration
            word_scores: dict, List of contour scores for each small word across all iterations
        """

        avg_scores = []
        word_scores = {word: [] for word in self.main_words}
        cp.random.seed(random_state) 

        gpu_embeddings = {}
        for main_word in self.main_words:
            unique_embedding_data = self.word_to_embeddings.get(main_word)
            if unique_embedding_data is not None and len(unique_embedding_data) >= 2:
                gpu_embeddings[main_word] = cp.asarray(unique_embedding_data, dtype=cp.float32)

        for _ in tqdm(range(n_bootstrap)):
            iter_word_scores = []
            for main_word, gpu_emb in gpu_embeddings.items():
                n_samples = len(gpu_emb)

                boot_indices = cp.random.choice(n_samples, n_samples, replace=True)
                boot_embeddings = gpu_emb[boot_indices]
               
                # Finding the best k
                best_sil = -1
                for k in range(2,16):
                    kmeans = cu_KMeans(n_clusters=k, random_state=random_state)
                    kmeans.fit(boot_embeddings)
                    labels = kmeans.labels_

                    sil_score = cu_silhouette_score(boot_embeddings, labels)
                    sil_score = float(sil_score)
                    if sil_score > best_sil:
                        best_sil = sil_score
                        
                iter_word_scores.append(best_sil)
                word_scores[main_word].append(best_sil)
            
            if iter_word_scores:
                avg_scores.append(np.mean(iter_word_scores))
        
        self.bootstrap_avg_scores = avg_scores
        self.bootstrap_word_scores = word_scores

        merged = {}
        try:
            existing = np.load(self.args.bootstrap_scores, allow_pickle=True)
            merged.update({k: existing[k] for k in existing.files})
        except FileNotFoundError:
            pass

        merged[f"{self.args.test_type}_bootstrap_avg_scores"] = np.array(self.bootstrap_avg_scores, dtype=float)
        merged[f"{self.args.test_type}_bootstrap_word_scores"] = np.array(
            [{k: v for k, v in self.bootstrap_word_scores.items()}],
            dtype=object
        )

        np.savez(self.args.bootstrap_scores, **merged)
        return avg_scores, word_scores
    
    def distance_matrix(self):
        checkpoint = torch.load(self.args.centroids_path + f"{self.args.test_type}_layer-1.pt", map_location=torch.device('cuda'))
        to_process = checkpoint.keys()
        compare_dict ={}
        for word in to_process:
            if word in self.main_words:
                mat = checkpoint[word].mean(dim=0)
                compare_dict[word] = mat

        ds = list(compare_dict.keys())
        n = len(ds)
        euclidean_matrix = np.zeros((n, n))

        for i in range(len(ds)):
            for j in range(len(ds)):
                if i == j:
                    euclidean_matrix[i, j] = 0.0 
                else:
                    vec_i = compare_dict[ds[i]]
                    vec_j = compare_dict[ds[j]]

                    euclidean_matrix[i, j] = torch.dist(vec_i, vec_j, p=2).item()

        ds_labels = [romkan.to_roma(word).replace('、', ',').replace('。', '.') for word in compare_dict.keys()]
        
        def plot_heatmap(matrix, title, save_path):
            df_cm = pd.DataFrame(matrix, index=ds_labels, columns=ds_labels)
            plt.figure(figsize=(12, 10))
            
            sns.heatmap(df_cm, annot=False, cmap="Blues", fmt=".4f", linewidths=0.5, vmin=0, vmax=100)
            plt.xlabel("Discourse Markers")
            plt.ylabel("Discourse Markers")
            plt.title(title)

            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        plot_heatmap(euclidean_matrix, "Distance Confusion Matrix (Euclidean)", self.args.distance_path)