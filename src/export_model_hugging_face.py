# coding=utf-8
# Exports the fine-tuned model.
# Usage: python export_model.py --checkpoint_dir path_to_checkpoint --output_dir path_to_save_model

from llmtuner import get_train_args, load_model_and_tokenizer
from huggingface_hub import login

def main():
    model_args, _, training_args, finetuning_args, _ = get_train_args()
    model, tokenizer = load_model_and_tokenizer(model_args, finetuning_args)
    login(token='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')
    model.push_to_hub("cuican1432/baichuan13b_dataphant", use_auth_token=True)


if __name__ == "__main__":
    main()
