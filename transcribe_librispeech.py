import os
from transformers import AutomaticSpeechRecognitionPipeline
import argparse
from typing import Tuple, Generator
import torch
import pandas as pd
import jiwer

LIBRISPEECH = os.environ.get('LIBRISPEECH')
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        '-o',
        help="Path of .csv file to save output to. Defaults to $model-$split.csv",
    )
    parser.add_argument(
        "--split",
        '-s',
        help="Split of Librispeech dataset to transcribe.",
        default="dev-clean",
    )
    parser.add_argument(
        "--model",
        "-m",
        help="Path to HuggingFace transformers model to use for transcription.",
        default="openai/whisper-large-v2"
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        help="Number of examples to process at a time.",
        default=32,
        type=int,
    )
    parser.add_argument(
        "--device",
        "-d",
        help="Device to run inference on.",
        default=DEVICE,
        type=lambda x: int(x) if x!='cpu' else x
    )
    return parser

def dataset_generator(libris_split_dir: str) -> Generator[Tuple[str, str], None, None]:
    """
    Generator object yielding tuples of `(recording_path,transcription)` for all examples
    in a LibriSpeech split directory.
    """
    for root, _, filenames in os.walk(libris_split_dir):
        transcription_files = [file for file in filenames if file.endswith('.trans.txt')]
        if not transcription_files:
            continue
        if len(transcription_files)>1:
            raise FileNotFoundError(f"Found multiple transcription files in directory {root}")
        transcription = os.path.join(root, transcription_files[0])
        with open(transcription) as fh:
            lines = fh.readlines()
            for line in lines:
                recording_stem = line.split()[0]
                recording_files = [file for file in filenames if file.startswith(recording_stem)]
                if len(recording_files)!=1:
                    raise FileNotFoundError(f"Expected 1 recording starting with {recording_stem}")
                recording = os.path.join(root, recording_files[0])
                transcription = " ".join(line.split()[1:])
                yield recording, transcription

def unzip(zipped_list):
    return list(zip(*zipped_list))

def transcribe_librispeech(args) -> int:
    """
    
    """
    librispeech_directory = os.path.join(LIBRISPEECH, args.split)
    print("Reading dataset...")
    example_tuples = [example for example in dataset_generator(librispeech_directory)]
    recordings, transcriptions = unzip(example_tuples)

    print("Transcribing...")
    pipe = AutomaticSpeechRecognitionPipeline(
        args.model,
        device=args.device
    )
    pipe_output = pipe(recordings, batch_size=args.batch_size)
    hypotheses = [output['text'] for output in pipe_output]

    df = pd.DataFrame({'reference': transcriptions, 'hypothesis': hypotheses})

    print("Computing WER...")
    df['wer'] = df.apply(
        lambda row: jiwer.wer(row['reference'], row['hypothesis'])
    )

    print("Saving output...")
    output_path = args.output
    if not output_path:
        model_basename = os.path.basename(args.model)
        output_path = f"{args.split}-{model_basename}.csv"
    df.to_csv(output_path)

    return 0

if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    exit(transcribe_librispeech(args))