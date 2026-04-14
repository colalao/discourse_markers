### fine-tuning
python train.py --language Japanese --pretrainModel gpt2

### get-embedding
- PCA
    - CUDA_VISIBLE_DEVICES=0 nohup python test.py --language English --pretrainModel gpt2 --test_type ft_no --pca > gpt2_ft_no.log 2>&1 &
    - CUDA_VISIBLE_DEVICES=5 python test.py --language Japanese --pretrainModel bert --test_type no_ft_no --pca

- NO_PCA
    - python test.py --language Japanese --pretrainModel gpt2 --test_type no_ft_no

### pretrainModel
- Llama3:
    - lightblue/suzume-llama-3-8B-multilingual
- Qwen3:
    - Qwen/Qwen3-8B-Base
- GPT2:
    - openai-community/gpt2
    - rinna/japanese-gpt2-medium
- Bert:
    - google-bert/bert-base-cased
    - tohoku-nlp/bert-base-japanese-v2


## Entropy[30%]

| Model | LLaMA3.2 | Mistral |
|---|---|---|
| no_emb | 1.9331| 1.8828 | 
| with_emb | 1.8153 | 1.7157 |

---

## Japanese

| Model | Bert | GPT2 | LLaMA3 | Qwen3 |
|---|---|---|---|---|
| no_ft_no | 0.208818 ± 0.006583 | 0.157326 ± 0.003548 | 0.256669 ± 0.009244 | 0.171602 ± 0.010683 |
| ft_no | 0.407318 ± 0.012891 | 0.288142 ± 0.009766 | 0.450449 ± 0.014332 | 0.452352 ± 0.067789 |
| no_ft_one | 0.178493 ± 0.004087 | 0.100720 ± 0.002954 | 0.178497 ± 0.002441 | 0.153607 ± 0.001657 |
| ft_one | 0.394423 ± 0.013163 | 0.273362 ± 0.008403 | 0.335048 ± 0.010218 | 0.262460 ± 0.013865 |
| no_ft_full | — | — | 0.317623 ± 0.021104 | 0.172486 ± 0.003584 |
| ft_full | — | — | 0.408378 ± 0.036720 | 0.180580 ± 0.012109 |