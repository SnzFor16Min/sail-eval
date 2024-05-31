#!/bin/bash
# Author: Jiaxin Guo <cs_guojiaxin@mail.scut.edu.cn>

CONDA_EXEC="conda"
ENV_NAME="sail_eval"
CUDA_VERSION="12.1"
SAIL_MODEL_PATH="<enter your path here>"
HF_CACHE_DIR=""
EVAL_DIR="../eval_outputs/"
MODEL_CONFIG_PATH="./configs/models/sail/sail_8b_instruct.py"

EVAL_DATASETS="mmlu_gen ceval_gen humaneval_gen mbpp_gen triviaqa_gen commonsenseqa_gen gsm8k_gen math_gen ARC_c_gen ARC_e_gen SuperGLUE_BoolQ_gen hellaswag_gen nq_gen obqa_gen piqa_gen siqa_gen"

CONDA_CMD="$CONDA_EXEC run --no-capture-output --name $ENV_NAME"
DATA_URL="https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-complete-20240207.zip"
EVAL_DIR="../eval_outputs/"
set -e

is_dataset_incomplete() {
  local DATASET_NAMES=("ceval" "commonsenseqa" "gsm8k" "hellaswag" "humaneval" "math" "mbpp" "mmlu" "nq" "piqa" "siqa")

  for name in "${DATASET_NAMES[@]}"; do
    if [ ! -d "./data/${name}/" ]; then
      return 0
    fi
  done
  return 1
}

get_gpu_id() {

  while true; do
    read -r -p "Enter GPU IDs (e.g., 0,2,3): " gpu_ids

    if [[ $gpu_ids =~ ^[0-9]+(,[0-9]+)*$ ]]; then
      break
    else
      echo "Invalid input. Please enter comma-separated GPU IDs (e.g., 0 or 0,1,2)." >&2
    fi
  done
  echo "$gpu_ids"
}

get_model_path() {

  while true; do
    read -r -p "Enter model directory path: " model_path

    if [ -d "$model_path" ]; then
      break
    else
      echo "Invalid directory path." >&2
    fi
  done
  echo "$model_path"
}

main() {
  cd "$(dirname "$0")"

  if [ ! -d ./opencompass/ ]; then
    while ! git clone --depth 1 https://github.com/open-compass/opencompass.git; do
      sleep 10
    done
  fi
  cd ./opencompass/

  if $CONDA_EXEC env list | grep -q "^$ENV_NAME\\b"; then
    echo "Will directly use the existing '$ENV_NAME' environment."
  else
    echo "Creating the '$ENV_NAME' ennvironment..."
    $CONDA_EXEC create --name $ENV_NAME \
      python=3.10 pytorch pytorch-cuda=$CUDA_VERSION faiss-gpu -c pytorch -c nvidia -y
    echo "Installing dependencies ..."
    $CONDA_CMD pip install -e .
  fi
  echo "Checking CUDA availability..."
  if ! $CONDA_CMD python -c "import torch; assert torch.cuda.is_available()"; then
    echo "PyTorch CUDA is not available!"
    exit 1
  fi

  if is_dataset_incomplete; then
    echo "Preparing datasets..."
    cd ..
    if [ ! -e "./$(basename $DATA_URL)" ]; then
      echo "Downloading datasets..."
      wget -c $DATA_URL
    fi
    unzip -n -d ./opencompass/ "$(basename $DATA_URL)"
    cd ./opencompass/data/
    find . -name "*.zip" -exec sh -c 'unzip -n "$1" && rm "$1"' sh {} \;
    cd ..
  fi

  if [ ! -d $SAIL_MODEL_PATH ]; then
    SAIL_MODEL_PATH=$(get_model_path)
  fi
  SAIL_MODEL_PATH=$(realpath "$SAIL_MODEL_PATH")
  if [ ! -e $MODEL_CONFIG_PATH ]; then
    cp -vr ../configs/ .
  fi
  sed -i "s|path='.*',$|path='$SAIL_MODEL_PATH',|" $MODEL_CONFIG_PATH
  cat $MODEL_CONFIG_PATH

  local gpu_ids=$(get_gpu_id)
  IFS=' ' read -ra datasets_array <<<"$EVAL_DATASETS"
  echo "Running evaluation script... (Check $(realpath $EVAL_DIR) for output)"
  CUDA_VISIBLE_DEVICES=$gpu_ids HF_HOME=$HF_CACHE_DIR HF_ENDPOINT=https://hf-mirror.com \
    $CONDA_CMD python -u run.py -r -w "$EVAL_DIR" --max-num-workers 8 \
    --models sail_8b_instruct --datasets "${datasets_array[@]}"
}

trap 'kill $(ps -f -u $(id -un) | egrep "openicl_infer.py|run.py" | grep -v grep | awk "{print \$2}"); exit 1' SIGINT
main
