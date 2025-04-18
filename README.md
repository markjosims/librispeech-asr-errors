# librispeech-asr-errors

Word error rate (WER) is the standard metric of evaluation for Automatic Speech Recognition (ASR) models. WER can be understood as the ratio of the number of edits need to correct an ASR hypothesis to the number of words in the reference (ground truth). WER is useful for gauging generally how well an ASR model can transcribe human speech, but it does not give any information about *what kinds* of errors an ASR model is prone to make. We can obtain richer information about the patterns of model errors through statistical analysis of model output on a standard benchmark.

This dataset consists of output for popular ASR models on [LibriSpeech dataset](https://www.openslr.org/12). Models used include Whisper, Wav2Vec 2.0 and HuBERT. The particular checkpoints used are as followS:
- Whisper large-v2: [whisper-large-v2](https://huggingface.co/openai/whisper-large-v2)
- Wav2Vec 2.0: [wav2vec2-large-960h-lv60](https://huggingface.co/facebook/wav2vec2-large-960h-lv60)
- HuBERT: [hubert-large-ls960-ft](https://huggingface.co/facebook/hubert-large-ls960-ft)

Models were evaluated on the **dev-clean** and **test-clean** partitions of LibriSpeech using greedy decoding. Reference and hypothesis text were normalized using the Whisper normalizer before computing WER. WER values were stored for each model, and `.csv` files were generated where each row corresponds to a word in the  reference string (i.e. files ending with `-references.csv`) or hypothesis string (files ending with `hypotheses.csv`) along with some data about the word (see below).

Data are stored in the `data.tar.gz` file. When unzipped the following directory will appear:

```
data/
├── wer.json  
├── dev/
│   ├── dev-clean-hubert-xlarge-ls960-ft-hypotheses.csv
│   ├── dev-clean-hubert-xlarge-ls960-ft-references.csv
│   ├── dev-clean-hubert-xlarge-ls960-ft-sentences.csv
│   ├── dev-clean-wav2vec2-large-960h-lv60-hypotheses.csv
│   ├── dev-clean-wav2vec2-large-960h-lv60-references.csv
│   ├── dev-clean-wav2vec2-large-960h-lv60-sentences.csv
│   ├── dev-clean-whisper-large-v2-hypotheses.csv
│   ├── dev-clean-whisper-large-v2-references.csv
│   └── dev-clean-whisper-large-v2-sentences.csv
└── test/
    ├── test-clean-hubert-xlarge-ls960-ft-hypotheses.csv
    ├── test-clean-hubert-xlarge-ls960-ft-references.csv
    ├── test-clean-hubert-xlarge-ls960-ft-sentences.csv
    ├── test-clean-wav2vec2-large-960h-lv60-hypotheses.csv
    ├── test-clean-wav2vec2-large-960h-lv60-references.csv
    ├── test-clean-wav2vec2-large-960h-lv60-sentences.csv
    ├── test-clean-whisper-large-v2-hypotheses.csv
    ├── test-clean-whisper-large-v2-references.csv
    └── test-clean-whisper-large-v2-sentences.csv
```

- `wer.json` contains the word error rate for each model on the dev-clean and test-clean datasets.
- The `-sentences.csv` files contain the following columns:
    - reference: The ground truth transcription
    - hypothesis: The predicted ASR transcription
    - reference_normalized: The ground truth transcription with Whisper string normalization applied
    - hypothesis_normalized: The predicted ASR transcription with Whisper string normalization applied
- The `-references.csv` and `-hypotheses.csv` files contain the following columns:
    - word: an individual word from a given reference or hypothesis sentence
    - prev_word: the preceding word, or `<s>` for beginning of sentence
    - next_word: the following word, or `</s>` for end of sentence
    - edit_type: 0 for hit (correct prediction), 1 for substitution, 2 for deletion (only applies to reference words), 3 for insertion (only applies to hypothesis words)
    - word_pos (prev_word_pos, next_word_pos): part of speech of word (or previous word or next word)
    - word_zipf_freq (prev_word_zipf_freq, next_word_zipf_freq): Zipf frequency of word (or previous word or next word)

Parts of speech were obtained with [nltk](https://www.nltk.org/) using `nltk.pos_tag`, and Zipf frequencies were obtained with [wordfreq](https://pypi.org/project/wordfreq/) using `wordfreq.zipf_frequency`.