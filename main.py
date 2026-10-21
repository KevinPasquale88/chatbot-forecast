import argparse
import subprocess
import sys

from chatbot_core import preload
from classification import training_classifier
from embedding import embeddings
from function import clean_output_dirs
from preprocessing import preprocessing_data

def parse_args():
    parser = argparse.ArgumentParser(description="Launch the clinical risk chatbot.")
    parser.add_argument(
        "--dataset", choices=["heart_disease", "diabetes130"], default="heart_disease",
        help="Clinical dataset to use (default: heart_disease)."
    )
    parser.add_argument(
        "--chatbot", choices=["telegram", "streamlit"], required=True,
        help="Chatbot front-end to launch, pre-set to --dataset."
    )
    return parser.parse_args()

def build_classifier(dataset):
    print(f"[INFO] Pulizia datas/{dataset}...")
    clean_output_dirs(dataset)

    print(f"[INFO] Lettura dataset '{dataset}'...")
    X, y = preprocessing_data(dataset=dataset)

    print(f"[INFO] Generazione embeddings ({len(X)} record)...")
    embeddings(X, y, dataset=dataset)

    print("[INFO] Addestramento classificatore...")
    training_classifier(dataset=dataset)

def launch_chatbot(kind, dataset):
    #build classfier if not already done
    build_classifier(dataset)

    preload(dataset)
    print("[INFO] Classificatore pronto.")

    if kind == "telegram":
        import bot_telegram
        bot_telegram.main(dataset=dataset)
    else:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "app_streamlit.py", "--", "--dataset", dataset],
            check=True,
        )

def main():
    #parsing arguments
    args = parse_args()
    #launching the chatbot with args
    launch_chatbot(args.chatbot, args.dataset)

if __name__ == "__main__":
    main()
