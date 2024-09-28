import os
from glob import glob
from multiprocessing import freeze_support

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

output_path = os.path.dirname(os.path.abspath(__file__))

mailabs_path = os.path.join(output_path, "MyTTSDataset/")
dataset_paths = glob(mailabs_path)
dataset_config = [
    BaseDatasetConfig(
        formatter="myformatter",
        meta_file_train='myset.csv',
        path=path,
        language='be',  # language code is the folder name
    )
    for path in dataset_paths
]

audio_config = VitsAudioConfig(
    sample_rate=22050,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    mel_fmin=0,
    mel_fmax=None,
)

vitsArgs = VitsArgs(
    use_language_embedding=True,
    embedded_language_dim=4,
    use_speaker_embedding=True,
    use_sdp=False,
)

config = VitsConfig(
    model_args=vitsArgs,
    audio=audio_config,
    run_name="vits_vctk",
    use_speaker_embedding=True,
    batch_size=32,
    eval_batch_size=16,
    batch_group_size=0,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    precompute_num_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=10,
    text_cleaner="multilingual_cleaners",
    use_phonemes=True,
    phoneme_language=None,
    phonemizer="multi_phonemizer",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=5,
    use_language_weighted_sampler=True,
    print_eval=True,
    mixed_precision=False,
    min_audio_len=audio_config.sample_rate,
    max_audio_len=audio_config.sample_rate * 10,
    output_path=output_path,
    datasets=dataset_config,
    test_sentences=[

        ["Я думаю, что этот стартап действительно удивительный.", "nikolaev", None, "ru"],
    ],
)

# force the convertion of the custom characters to a config attribute
config.from_dict(config.to_dict())

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

# load training samples
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=0.010416666666666666,
)

# init speaker manager for multi-speaker training
# it maps speaker-id to speaker-name in the model and data-loader
speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
config.model_args.num_speakers = speaker_manager.num_speakers

language_manager = LanguageManager(config=config)
config.model_args.num_languages = language_manager.num_languages

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

# init model
model = Vits(config, ap, tokenizer, speaker_manager, language_manager)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)

if __name__ == '__main__':
    freeze_support()
    trainer.fit()