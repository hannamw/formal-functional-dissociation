python find_circuit.py --model mistralai/Mistral-7B-v0.3
python overlap.py --model mistralai/Mistral-7B-v0.3
python cross-task-faithfulness.py --model mistralai/Mistral-7B-v0.3

python find_circuit.py --model allenai/OLMo-7B-hf
python overlap.py --model allenai/OLMo-7B-hf
python cross-task-faithfulness.py --model allenai/OLMo-7B-hf

python find_circuit.py --model Qwen/Qwen2.5-7B
python overlap.py --model Qwen/Qwen2.5-7B
python cross-task-faithfulness.py --model Qwen/Qwen2.5-7B

python find_circuit.py --model meta-llama/Meta-Llama-3-8B
python overlap.py --model meta-llama/Meta-Llama-3-8B
python cross-task-faithfulness.py --model meta-llama/Meta-Llama-3-8B

python find_circuit.py --model google/gemma-2-2b
python overlap.py --model google/gemma-2-2b
python cross-task-faithfulness.py --model google/gemma-2-2b