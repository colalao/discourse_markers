### fine-tuning
python train.py --language Japanese --pretrainModel gpt2

### get-embedding
- PCA
    - CUDA_VISIBLE_DEVICES=0 nohup python test.py --language English --pretrainModel gpt2 --test_type ft_no --pca > gpt2_ft_no.log 2>&1 &
    - CUDA_VISIBLE_DEVICES=5 python test.py --language Japanese --pretrainModel bert --test_type no_ft_no --pca

- NO_PCA
    - python test.py --language Japanese --pretrainModel gpt2 --test_type no_ft_no

### inference
- metric evaluation
    - python infer.py --language English --pretrainModel llama3 --test_type ft_one --infer_ratio 10
    - python infer.py --language English --pretrainModel llama3 --test_type no_ft_one --infer_ratio 10

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
