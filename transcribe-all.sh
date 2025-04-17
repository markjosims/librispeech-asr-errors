LIBRISPEECH=~/datasets/LibriSpeech
W2V2=facebook/wav2vec2-large-960h-lv60
HuBERT=facebook/hubert-xlarge-ls960-ft
WHISPER=openai/whisper-large-v2
for split in dev-clean test-clean; do
    for model in $W2V2 $HuBERT $WHISPER; do
        python transcribe_librispeech.py -m $model -s $split
    done
done