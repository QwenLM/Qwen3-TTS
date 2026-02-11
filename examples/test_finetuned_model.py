# coding=utf-8
# Test script for finetuned Qwen3-TTS model
import argparse
import os
import json
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel


def patch_config_for_inference(model_path: str):
    """
    Patch config.json to set tts_model_type to 'base' for inference.
    The finetuning script sets it to 'new_language' but inference requires 'base'.
    Skips if config.json doesn't exist (e.g., HuggingFace model ID).
    """
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    if config.get("tts_model_type") != "base":
        print(f"Patching config: tts_model_type '{config.get('tts_model_type')}' -> 'base'")
        config["tts_model_type"] = "base"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="Test finetuned Qwen3-TTS model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to finetuned model")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output file path for generated audio (e.g., output.wav)")
    parser.add_argument("--ref_audio", type=str, 
                        default="/mnt/data/meaning-corpora/cro-scrape-v2-processed-v3/059/host_dne_na_morave_petr_nekuza_z_technickeho_muzea_brno/host_dne_na_morave_petr_nekuza_z_technickeho_muzea_brno=165=170.flac",
                        help="Reference audio path for voice cloning")
    parser.add_argument("--text", type=str, 
                        default="Ahoj, jak se máš? Mě by to vadilo, kdyby se ti něco stalo. Rozhodně bych ti nerad ublížil ty česká kurvo.",
                        help="Text to synthesize")
    args = parser.parse_args()

    device = "cuda:0"

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Patch config for inference (finetuning sets 'new_language', but inference needs 'base')
    patch_config_for_inference(args.model_path)

    print(f"Loading finetuned model from: {args.model_path}")
    tts = Qwen3TTSModel.from_pretrained(
        args.model_path,
        device_map=device,
        dtype=torch.bfloat16,
    )
    
    print("Model loaded successfully!")

    common_gen_kwargs = dict(
        max_new_tokens=2048,
        do_sample=True,
        top_k=50,
        top_p=1.0,
        temperature=0.9,
        repetition_penalty=1.05,
        subtalker_dosample=True,
        subtalker_top_k=16,
        subtalker_top_p=0.95,
        subtalker_temperature=0.7,
    )

    print(f"\nGenerating audio for: {args.text[:50]}...")
    wavs, sr = tts.generate_voice_clone(
        text=args.text,
        language=None,
        ref_audio=args.ref_audio,
        ref_text=None,
        x_vector_only_mode=True,
        non_streaming_mode=True,
        **common_gen_kwargs,
    )

    sf.write(args.output_file, wavs[0], sr)
    print(f"Saved to: {args.output_file}")


if __name__ == "__main__":
    main()
