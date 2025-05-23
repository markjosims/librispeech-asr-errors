import os
from transformers import pipeline, WhisperTokenizer
import argparse
from typing import Tuple, Generator, List, Dict, Any
import torch
import pandas as pd
import jiwer
from tqdm import tqdm
from datasets import Dataset, Audio
import math
import numpy as np
from nltk.util import ngrams
from nltk import pos_tag
from wordfreq import zipf_frequency
import json

LIBRISPEECH = os.environ.get('LIBRISPEECH')
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

def init_parser():
    parser = argparse.ArgumentParser()
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
        "--device",
        "-d",
        help="Device to run inference on.",
        default=DEVICE,
        type=lambda x: int(x) if x!='cpu' else x
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        help="Number of examples to process at a time.",
        default=32,
        type=int,
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
    return [list(unzipped) for unzipped in zip(*zipped_list)]

def dataloader(dataset: Dataset, batch_size: int) -> Generator[Dataset, None, None]:
    indices = range(len(dataset))
    for start_index in range(0, len(dataset), batch_size):
        end_index = start_index+batch_size
        yield dataset.select(indices[start_index:end_index])

def get_audio_dataset(recordings: List[str]) -> Dataset:
    ds = Dataset.from_dict({'audio': recordings})
    ds = ds.cast_column('audio', Audio(sampling_rate=16_000))
    return ds

def get_pipe_kwargs(model: str) -> Dict[str, Any]:
    if 'whisper' in model:
        return {
            'return_timestamps': True,
            'generate_kwargs': {'language': 'english'},
        }
    # HuBERT and wav2vec2 are CTC models
    return {
        'chunk_length_s': 10,
        'stride_length_s': (4, 2),
    }

def whisper_normalize(sentences: List[str]) -> List[str]:
    tokenizer = WhisperTokenizer.from_pretrained('openai/whisper-large-v2')
    return [tokenizer.normalize(sent) for sent in sentences]

def get_edits(reference_list, hypothesis_list):
    """
    Given reference and hypothesis strs calculate word edits using `jiwer`.
    Then return a dictionary with two arrays, 'reference_edits' and 'hypothesis_edits',
    of lengths *n* and *m*, n=num_words(reference) and m=num_words(hypothesis).
    'reference_edits' has a 0 for every hit, a 1 for every substitution and a 2 for every deletion.
    'hypothesis_edits' has a 0 for every hit, a 1 for every substitution and a 3 for every insertion.
    """
    processed_words = jiwer.process_words(reference_list, hypothesis_list)
    alignments = processed_words.alignments
    ref_edit_list = []
    hyp_edit_list = []
    for reference, hypothesis, alignment in zip(reference_list, hypothesis_list, alignments):
        hyp_edits = np.zeros(len(hypothesis.split()))
        ref_edits = np.zeros(len(reference.split()))
        for aligned_chunk in alignment:
            edit_type = aligned_chunk.type
            hyp_start, hyp_end = aligned_chunk.hyp_start_idx, aligned_chunk.hyp_end_idx
            ref_start, ref_end = aligned_chunk.ref_start_idx, aligned_chunk.ref_end_idx

            if edit_type == 'equal':
                # do nothing, as hits are 0
                pass
            elif edit_type == 'substitute':
                hyp_edits[hyp_start:hyp_end]=1
                ref_edits[ref_start:ref_end]=1
            elif edit_type == 'delete':
                ref_edits[ref_start:ref_end]=2
            else:
                # edit_type == 'insert'
                hyp_edits[hyp_start:hyp_end]=3
        ref_edit_list.append(ref_edits)
        hyp_edit_list.append(hyp_edits)
    return ref_edit_list, hyp_edit_list

def get_3grams(sent: str) -> List[Tuple[str,str,str]]:
    """
    Converts sentence into list of 3-grams.
    Trims first and last 3-grams, so that len(3-grams)=num_words(sent)
    and every 3-gram is centered on a word,
    otherwise 3-grams will include (<s>, <s>, WORD) and (WORD, </s>, </s>).
    """
    sent_3grams = list(
        ngrams(
            sequence=sent.split(),
            n=3,
            pad_left=True,
            pad_right=True,
            left_pad_symbol='<s>',
            right_pad_symbol='</s>',
        )
    )
    sent_3grams_trimmed = sent_3grams[1:-1]
    return sent_3grams_trimmed

def get_word_df(sentences, edits):
    sent_3grams = [get_3grams(sent) for sent in sentences]
    rows = {
        'word': [],
        'prev_word': [],
        'next_word': [],
        'edit_type': [],
        'sentence': [],
    }
    for sent_edit, sent_3gram, sent in zip(edits, sent_3grams, sentences):
        rows['edit_type'].extend(sent_edit)
        prev_words, words, next_words = unzip(sent_3gram)
        rows['prev_word'].extend(prev_words)
        rows['word'].extend(words)
        rows['next_word'].extend(next_words)
        rows['sentence'].extend([sentences]*len(sent_edit))
    df = pd.DataFrame(rows)
    return df

def add_pos(word_df):
    for word_col in ['word', 'prev_word', 'next_word']:
        pos_tuples = pos_tag(word_df[word_col])
        pos = [t[1] for t in pos_tuples]
        word_df[word_col+'_pos'] = pos
        
        pad_token_mask = word_df[word_col].isin(['<s>', '</s>'])
        word_df.loc[pad_token_mask, word_col+'_pos'] = word_df.loc[pad_token_mask, word_col]
    return word_df

def add_wordfreq(word_df):
    for word_col in ['word', 'prev_word', 'next_word']:
        word_df[word_col+'_zipf_freq'] = word_df[word_col].apply(lambda w: zipf_frequency(w, 'en'))
        pad_token_mask = word_df[word_col].isin(['<s>', '</s>'])
        word_df.loc[pad_token_mask, word_col+'_zipf_freq'] = -1
    return word_df

def transcribe_librispeech(args) -> int:
    """
    
    """
    librispeech_directory = os.path.join(LIBRISPEECH, args.split)
    print("Reading dataset...")
    example_tuples = [example for example in dataset_generator(librispeech_directory)]
    recordings, transcriptions = unzip(example_tuples)
    ds = get_audio_dataset(recordings)

    print(f"Loading ASR pipeline for {args.model}...")
    pipe = pipeline(
        "automatic-speech-recognition",
        args.model,
        device=args.device
    )
    print("Transcribing...")
    hypotheses = []
    num_batches = math.ceil(len(ds)/args.batch_size)
    pipe_kwargs = get_pipe_kwargs(args.model)
    for batch in tqdm(dataloader(ds, args.batch_size), total=num_batches):
        pipe_output = pipe(batch['audio'], **pipe_kwargs)
        hypotheses.extend(output['text'] for output in pipe_output)

    print("Computing WER...")
    refs_nmzd = whisper_normalize(transcriptions)
    hyps_nmzd = whisper_normalize(hypotheses)
    ref_edits, hyp_edits = get_edits(refs_nmzd, hyps_nmzd)
    overall_wer = jiwer.wer(refs_nmzd, hyps_nmzd)

    print("Making dataframe for sentences...")
    sentence_df = pd.DataFrame({
        "reference": transcriptions,
        "hypothesis": hypotheses,
        "reference_normalized": refs_nmzd,
        "hypothesis_normalized": hyps_nmzd,
    })

    print("Making dataframe for each word in references...")
    ref_df = get_word_df(refs_nmzd, ref_edits)
    hyp_df = get_word_df(hyps_nmzd, hyp_edits)

    print("Tagging for part of speech and Zipf frequency...")
    ref_df = add_pos(ref_df)
    ref_df = add_wordfreq(ref_df)
    
    hyp_df = add_pos(hyp_df)
    hyp_df = add_wordfreq(hyp_df)

    print("Saving output...")
    model_basename = os.path.basename(args.model)
    os.makedirs('data', exist_ok=True)
    output_stem = os.path.join("data" ,f"{args.split}-{model_basename}")
    sentence_df.to_csv(output_stem+'-sentences.csv')
    ref_df.to_csv(output_stem+'-references.csv')
    hyp_df.to_csv(output_stem+'-hypotheses.csv')

    wer_json = {
        'model': args.model,
        'split': args.split,
        'wer': overall_wer,
    }
    wer_json_path = os.path.join('data', 'wer.json')
    wer_json_series = []
    if os.path.exists(wer_json_path):
        with open(wer_json_path) as f:
            wer_json_series = json.load(f)
    wer_json_series.append(wer_json)
    with open(wer_json_path, 'w') as f:
        json.dump(wer_json_series, f)

    return 0

if __name__ == '__main__':
    parser = init_parser()
    args = parser.parse_args()
    exit(transcribe_librispeech(args))