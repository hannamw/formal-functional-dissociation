
python find_circuit_node.py --model mistralai/Mistral-7B-v0.3
python overlap_virtual.py --model mistralai/Mistral-7B-v0.3 --level node
python cross-task-faithfulness.py --model mistralai/Mistral-7B-v0.3 --level node

python find_circuit_node.py --model allenai/OLMo-7B-hf
python overlap_virtual.py --model allenai/OLMo-7B-hf --level node
python cross-task-faithfulness.py --model allenai/OLMo-7B-hf --level node

python find_circuit_node.py --model Qwen/Qwen2.5-7B
python overlap_virtual.py --model Qwen/Qwen2.5-7B --level node
python cross-task-faithfulness.py --model Qwen/Qwen2.5-7B --level node

python find_circuit_node.py --model meta-llama/Meta-Llama-3-8B
python overlap_virtual.py --model meta-llama/Meta-Llama-3-8B --level node
python cross-task-faithfulness.py --model meta-llama/Meta-Llama-3-8B --level node

python find_circuit_node.py --model google/gemma-2-2b
python overlap_virtual.py --model google/gemma-2-2b --level node
python cross-task-faithfulness.py --model google/gemma-2-2b --level node