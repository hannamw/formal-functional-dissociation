declare -a tasks=('ioi' 'colored-objects' 'entity-tracking' 'greater-than-multitoken' 'fact-retrieval-comma' 'fact-retrieval-rev' 'gendered-pronoun' 'sva' 'hypernymy-comma' 'npi')

## now loop through the above array
for task in "${tasks[@]}"
do
   cd $task
   python create_dataset.py --model "mistralai/Mistral-7B-v0.3"
   cd ..
done