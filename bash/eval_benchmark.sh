
START=0
END=360
STEP=10
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
BASE_URL="http://localhost:8000/v1/completions"
NUM_CONCURRENT=1
MAX_RETRIES=3
TOKENIZED_REQUESTS=False

# Use the wrapper script that registers the model
for THETA in $(seq $START $STEP $END); do
    python scripts/lm_eval_steering.py --model steering-completions \
    --tasks tinyGSM8k \
    --model_args model=$MODEL_ID,tokenizer=$MODEL_ID,base_url=$BASE_URL,theta=$THETA,num_concurrent=$NUM_CONCURRENT,max_retries=$MAX_RETRIES,tokenized_requests=$TOKENIZED_REQUESTS \
    --output_path results/tinyGSM8k_theta_$THETA.json
done

# Example with specific theta value
python scripts/lm_eval_steering.py --model steering-completions --tasks tinyGSM8k \
  --model_args model=Qwen/Qwen2.5-7B-Instruct,base_url=http://localhost:8000/v1/completions,theta=20.0,\
num_concurrent=1,max_retries=3,tokenized_requests=False,do_sample=False \
  --output_path results/tinyGSM8k_theta_20.json