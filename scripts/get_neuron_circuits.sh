python find_circuit_node.py --model mistralai/Mistral-7B-v0.3 --neuron
python overlap_virtual.py --model mistralai/Mistral-7B-v0.3 --level neuron
python cross-task-faithfulness.py --model mistralai/Mistral-7B-v0.3 --level neuron

python find_circuit_node.py --model allenai/OLMo-7B-hf --neuron
python overlap_virtual.py --model allenai/OLMo-7B-hf --level neuron
python cross-task-faithfulness.py --model allenai/OLMo-7B-hf --level neuron

python find_circuit_node.py --model Qwen/Qwen2.5-7B --neuron
python overlap_virtual.py --model Qwen/Qwen2.5-7B --level neuron
python cross-task-faithfulness.py --model Qwen/Qwen2.5-7B --level neuron

python find_circuit_node.py --model meta-llama/Meta-Llama-3-8B --neuron
python overlap_virtual.py --model meta-llama/Meta-Llama-3-8B --level neuron
python cross-task-faithfulness.py --model meta-llama/Meta-Llama-3-8B --level neuron

python find_circuit_node.py --model google/gemma-2-2b --neuron
python overlap_virtual.py --model google/gemma-2-2b --level neuron
python cross-task-faithfulness.py --model google/gemma-2-2b --level neuron