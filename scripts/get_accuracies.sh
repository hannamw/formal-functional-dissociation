export CUDA_VISIBLE_DEVICES=0

python get_accuracy.py --model mistralai/Mistral-7B-v0.3

python get_accuracy.py --model allenai/OLMo-7B-hf

python get_accuracy.py --model Qwen/Qwen2.5-7B

python get_accuracy.py --model meta-llama/Meta-Llama-3-8B

python get_accuracy.py --model google/gemma-2-2b