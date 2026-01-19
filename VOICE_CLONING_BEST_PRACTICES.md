# Fish Speech Voice Cloning & TTS Best Practices

A comprehensive research guide for getting the best results from Fish Speech (FishAudio-S1) voice cloning and text-to-speech synthesis.

## Table of Contents

1. [Overview](#overview)
2. [Voice Cloning Best Practices](#voice-cloning-best-practices)
3. [Training and Fine-Tuning](#training-and-fine-tuning)
4. [Achieving Consistent Tone](#achieving-consistent-tone)
5. [Emotional Performance Control](#emotional-performance-control)
6. [Iteration Workflow](#iteration-workflow)
7. [Model Integration and Compatibility](#model-integration-and-compatibility)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Advanced Tips and Tricks](#advanced-tips-and-tricks)

---

## Overview

Fish Speech (FishAudio-S1) is a state-of-the-art open-source TTS and voice cloning model that focuses on natural, expressive, and emotionally rich speech synthesis. This guide consolidates best practices, workflows, and techniques to maximize the quality of your voice cloning and TTS results.

**Key Features:**
- Zero-shot voice cloning with 10-30 seconds of audio
- 64+ emotion markers for fine-grained control
- Multilingual support (13+ languages)
- Dual-Autoregressive architecture for consistency
- #1 ranking on TTS-Arena2 benchmark

---

## Voice Cloning Best Practices

### Recording Environment

**Select a Quiet Environment:**
- **Best locations:** Bedrooms with curtains/carpets, parked cars, quiet offices
- **Avoid:** Open windows, running appliances, background conversations, echoey spaces
- **Why:** Background noise and reverb reduce clone accuracy and naturalness

### Recording Equipment

**Use Appropriate Hardware:**
- **Best:** USB microphone or quality gaming headset
- **Acceptable:** Phone voice recorder app (modern smartphones)
- **Technique:**
  - Hold microphone steady, about a hand's width from your mouth
  - Maintain consistent distance throughout recording
  - Speak at steady, natural volume (avoid shouting or whispering)

### Recording Length and Content

**Optimal Sample Specifications:**
- **Minimum:** 10 seconds (basic voice clone)
- **Recommended:** 20-30 seconds (studio-quality accuracy)
- **Ideal:** 2-3 clips of 15-20 seconds each, forming complete paragraphs
- **Content suggestions:**
  - Talk about your interests or daily activities
  - Read a descriptive paragraph
  - Use natural, varied language with diverse phonemes

### Recording Quality Guidelines

**Consistency is Key:**
- Keep tone, volume, and emotion **even** throughout recording
- Speak naturally and steadily
- Pause slightly (half a second) between sentences
- Only one person should speak in each recording
- No overlapping voices or background music
- Avoid large emotional swings in reference audio

**Audio Processing:**
- Apply loudness normalization to your dataset
- Use `fish-audio-preprocess` for best results:
  ```bash
  fap loudness-norm data-raw data --clean
  ```

### Upload and Testing Workflow

1. **Upload Process:**
   - Log in at [fish.audio](https://fish.audio/)
   - Go to 'Create Voice'
   - Upload your recording
   - Name your voice
   - Wait for processing (typically seconds to a minute)

2. **Testing & Iteration:**
   - Generate various speech samples to test quality
   - If unsatisfactory, re-record in quieter space or use longer/clearer sample
   - Test with different emotions and scenarios

### Legal and Ethical Guidelines

- ⚠️ **Only clone voices you have explicit permission to use**
- Do not use celebrity, public figures, or any online voices without documented consent
- Respect copyright and privacy laws in your jurisdiction

---

## Training and Fine-Tuning

### Can You Train on a Voice?

**Yes!** Fish Speech supports fine-tuning to improve performance on specific voices or datasets. In the current version, you only need to fine-tune the **LLAMA** part of the model.

### Dataset Preparation

**Directory Structure:**
```
data/
├── SPK1/
│   ├── 21.15-26.44.lab
│   ├── 21.15-26.44.mp3
│   ├── 27.51-29.98.lab
│   ├── 27.51-29.98.mp3
│   └── ...
└── SPK2/
    ├── 38.79-40.85.lab
    ├── 38.79-40.85.mp3
    └── ...
```

**File Requirements:**
- **Audio formats:** `.mp3`, `.wav`, or `.flac`
- **Label files:** `.lab` extension containing only the transcription text
- **Example:** If `hi.mp3` says "Hello, goodbye," then `hi.lab` contains: `Hello, goodbye.`

### Training Workflow for Emotional Performance

**1. Dataset Diversity:**
- Include recordings with various emotional deliveries (happy, sad, angry, surprised)
- If source data is monotone, augment with external emotional datasets
- Balance data to avoid losing speaker identity while gaining emotional range

**2. Emotion Annotation:**
- Use Fish Speech emotion markers in transcripts:
  ```
  (happy) What a beautiful day!
  (sad) I'm sorry to hear that.
  (excited) This is amazing!
  ```
- Support for 64+ emotion tags (see [Emotional Performance Control](#emotional-performance-control))

**3. Fine-Tuning Steps:**

**Step 1: Download Model Weights**
```bash
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
```

**Step 2: Extract Semantic Tokens**
```bash
python tools/vqgan/extract_vq.py data \
    --num-workers 1 --batch-size 16 \
    --config-name "modded_dac_vq" \
    --checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth"
```

**Step 3: Pack Dataset into Protobuf**
```bash
python tools/llama/build_dataset.py \
    --input "data" \
    --output "data/protos" \
    --text-extension .lab \
    --num-workers 16
```

**Step 4: Fine-Tune with LoRA**
```bash
python fish_speech/train.py --config-name text2semantic_finetune \
    project=$project \
    +lora@model.model.lora_config=r_8_alpha_16
```

**Adjust training parameters** in `fish_speech/configs/text2semantic_finetune.yaml`:
- `batch_size`
- `gradient_accumulation_steps`
- Epochs (typically 10-20 for large datasets)

**Step 5: Merge LoRA Weights**
```bash
python tools/llama/merge_lora.py \
    --lora-config r_8_alpha_16 \
    --base-weight checkpoints/openaudio-s1-mini \
    --lora-weight results/$project/checkpoints/step_000000010.ckpt \
    --output checkpoints/openaudio-s1-mini-yth-lora/
```

**💡 Training Tips:**
- Use earliest checkpoint that meets requirements (better OOD performance)
- Model learns speech patterns, not timbre—still need prompts for timbre stability
- Increase training steps to learn timbre, but risk overfitting
- For Windows: use `trainer.strategy.process_group_backend=gloo` to avoid nccl issues

---

## Achieving Consistent Tone

Fish Speech employs several architectural innovations for tone consistency:

### 1. Dual-Autoregressive (Dual-AR) Architecture

**How it works:**
- **Slow Transformer:** Handles high-level linguistic and prosodic structure
- **Fast Transformer:** Adds detailed acoustic information
- **Result:** Decouples global prosody from local acoustic detail, stabilizing tone

### 2. Firefly-GAN Vocoder

**Advanced Features:**
- Grouped Scalar Vector Quantization (GSVQ)
- Near 100% codebook utilization
- Deep convolutional layers for expressive preservation
- **Result:** Reduces artifacts, improves fidelity across languages

### 3. Massive Training Data

- Trained on 720,000+ hours of multilingual audio
- Diverse accents, speakers, and scenarios
- **Result:** High generalization and consistency

### Practical Tips for Consistent Tone

**1. Use Reference Audio Effectively:**
```bash
# Generate VQ tokens from reference audio
python fish_speech/models/dac/inference.py \
    -i "ref_audio_name.wav" \
    --checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth"
```

**2. Leverage Prompt Engineering:**
```bash
# Generate with reference
python fish_speech/models/text2semantic/inference.py \
    --text "The text you want to convert" \
    --prompt-text "Your reference text" \
    --prompt-tokens "fake.npy" \
    --compile
```

**3. Save Reference Profiles:**
- Store label files and reference audio in `references/` folder
- Structure: `references/<voice_id>/sample.lab` and `sample.wav`
- Quick access in WebUI

**4. Fine-Tune on Target Voice:**
- By default, model learns speech patterns, not timbre
- Use prompts for timbre stability
- Increase training steps to learn timbre (monitor for overfitting)

---

## Emotional Performance Control

Fish Speech supports **fine-grained emotion control** through text markers—a unique feature among TTS models.

### Basic Emotions (26 tags)
```
(angry) (sad) (excited) (surprised) (satisfied) (delighted) 
(scared) (worried) (upset) (nervous) (frustrated) (depressed)
(empathetic) (embarrassed) (disgusted) (moved) (proud) (relaxed)
(grateful) (confident) (interested) (curious) (confused) (joyful)
```

### Advanced Emotions (25 tags)
```
(disdainful) (unhappy) (anxious) (hysterical) (indifferent) 
(impatient) (guilty) (scornful) (panicked) (furious) (reluctant)
(keen) (disapproving) (negative) (denying) (astonished) (serious)
(sarcastic) (conciliative) (comforting) (sincere) (sneering)
(hesitating) (yielding) (painful) (awkward) (amused)
```

### Tone Markers (5 tags)
```
(in a hurry tone) (shouting) (screaming) (whispering) (soft tone)
```

### Special Audio Effects (10 tags)
```
(laughing) (chuckling) (sobbing) (crying loudly) (sighing) (panting)
(groaning) (crowd laughing) (background laughter) (audience laughing)
```

### Usage Examples

**Simple Emotion:**
```
(happy) What a wonderful day we're having!
```

**Multiple Emotions in Sequence:**
```
(excited) I can't believe it! (surprised) Wait, really? (joyful) This is amazing!
```

**Combining Emotions with Tone:**
```
(whispering) (nervous) Don't let them hear us.
```

**Laughter Control:**
```
Ha, ha, ha! (chuckling) That's hilarious!
```

### Language Support for Emotions

**Supported Languages:**
English, Chinese, Japanese, German, French, Spanish, Korean, Arabic, Russian, Dutch, Italian, Polish, Portuguese

### Tips for Better Emotional Performance

1. **Dataset Quality:** Train with emotionally diverse samples
2. **Augmentation:** Blend external emotional datasets carefully
3. **Balance:** Maintain speaker identity while adding emotional range
4. **Testing:** Evaluate both objectively (MOS) and subjectively
5. **Iteration:** Refine emotion tags and training if output lacks expression

---

## Iteration Workflow

### End-to-End Process for Refining Results

#### Phase 1: Environment Setup
```bash
# Clone repository
git clone https://github.com/fishaudio/fish-speech.git
cd fish-speech

# Set up environment (recommended: conda/venv)
conda create -n fish-speech python=3.10
conda activate fish-speech

# Install dependencies
pip install -e .

# Download model weights
huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
```

#### Phase 2: Data Preparation
1. **Organize Dataset:**
   - Structure audio files and labels
   - Ensure consistent sampling rates (16kHz or 24kHz)
   - Remove long silences

2. **Preprocess:**
   ```bash
   # Loudness normalization
   fap loudness-norm data-raw data --clean
   ```

3. **Extract Features:**
   ```bash
   python tools/vqgan/extract_vq.py data \
       --num-workers 1 --batch-size 16 \
       --config-name "modded_dac_vq" \
       --checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth"
   ```

#### Phase 3: Training & Validation
1. **Configure Training:**
   - Edit `fish_speech/configs/text2semantic_finetune.yaml`
   - Adjust batch size, epochs, validation intervals

2. **Train Model:**
   ```bash
   python fish_speech/train.py --config-name text2semantic_finetune \
       project=my_voice \
       +lora@model.model.lora_config=r_8_alpha_16
   ```

3. **Monitor Progress:**
   - Set low `val_check_interval` for early issue detection
   - Watch for overfitting

#### Phase 4: Inference & Testing
1. **Generate Semantic Tokens:**
   ```bash
   python fish_speech/models/text2semantic/inference.py \
       --text "Test text" \
       --prompt-text "Reference text" \
       --prompt-tokens "fake.npy" \
       --compile
   ```

2. **Synthesize Audio:**
   ```bash
   python fish_speech/models/dac/inference.py \
       -i "codes_0.npy"
   ```

3. **Evaluate:**
   - Listen to output
   - Test with various emotions and scenarios
   - Collect feedback (qualitative and quantitative)

#### Phase 5: Refinement Loop
1. **Analyze Issues:**
   - Robotic output? → Longer, clearer recordings
   - Lack of likeness? → More natural, varied phonemes
   - Poor quality? → Quieter space, better equipment

2. **Adjust and Retry:**
   - Modify prompts
   - Refine dataset
   - Adjust training parameters
   - Retrain if necessary

3. **Automated Testing:**
   - Batch synthesis for large-scale testing
   - MOS (Mean Opinion Score) evaluations

---

## Model Integration and Compatibility

### Using Other Models with Fish Speech

**Native Compatibility:**
- Fish Speech API supports multiple internal model versions:
  - `s1` (FishAudio-S1/OpenAudio S1 - flagship)
  - `speech-1.5`, `speech-1.6`
  - `v1`, `v2`, `v3-turbo`, `v3-hd`

**Switching Models:**
```bash
# Via API (set model parameter)
curl -X POST https://fishspeech.net/api/open/tts \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -d '{
    "reference_id": "your_model_id",
    "text": "Text content",
    "version": "s1",
    "format": "mp3"
  }' --output output.mp3
```

**Third-Party Model Integration:**
- Fish Speech API is designed for Fish Speech models only
- **Not directly compatible** with other TTS systems (TortoiseTTS, Coqui, ElevenLabs)
- **Custom integration possible** by extending the codebase to build multi-backend solution
- Open-source nature allows for customization

### API Endpoints

**RESTful API:**
```
POST https://api.fish.audio/v1/tts
Headers:
  Authorization: Bearer YOUR_API_KEY
  Content-Type: application/json
  model: s1

Body:
{
  "text": "Hello! Welcome to Fish Audio."
}
```

**Other Endpoints:**
- `/v1/asr` - Speech-to-text
- Model management (list, get, create, update, delete)
- API credit and package status

**Request Parameters:**
- `reference_id` - Voice Model ID for cloning
- `text` - Text to synthesize
- `speed`, `volume` - Playback controls
- `version` & `format` - Model and output format
- `emotion`, `language` - For V3 models
- `cache` - Direct audio or downloadable link

### Local Deployment Options

**WebUI:**
```bash
python -m tools.run_webui \
    --llama-checkpoint-path "checkpoints/openaudio-s1-mini" \
    --decoder-checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth" \
    --decoder-config-name modded_dac_vq
```

**API Server:**
```bash
python -m tools.api_server \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path "checkpoints/openaudio-s1-mini" \
    --decoder-checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth" \
    --decoder-config-name modded_dac_vq
```

**Docker:**
```bash
# WebUI with CUDA
docker run -d \
    --name fish-speech-webui \
    --gpus all \
    -p 7860:7860 \
    -v ./checkpoints:/app/checkpoints \
    -v ./references:/app/references \
    -e COMPILE=1 \
    fishaudio/fish-speech:latest-webui-cuda

# API Server with CUDA
docker run -d \
    --name fish-speech-server \
    --gpus all \
    -p 8080:8080 \
    -v ./checkpoints:/app/checkpoints \
    -e COMPILE=1 \
    fishaudio/fish-speech:latest-server-cuda
```

**Performance Optimization:**
- Add `--compile` flag for ~10x faster inference (CUDA only)
- Uses torch.compile to fuse CUDA kernels
- RTX 4090: ~15 tokens/s → ~150 tokens/s

---

## Troubleshooting Guide

### Common Issues and Solutions

#### Robotic or Unnatural Output

**Symptoms:** Speech sounds mechanical, lacks emotion, or has strange prosody

**Solutions:**
1. Use longer, clearer reference recordings (20-30 seconds)
2. Ensure reference has natural phrasing and varied phonemes
3. Try different reference samples
4. Add emotion markers to text
5. Check that only one speaker is in reference audio

#### Lack of Voice Likeness

**Symptoms:** Generated voice doesn't match target

**Solutions:**
1. Use higher quality reference audio (quieter environment, better mic)
2. Ensure consistent tone and volume in reference
3. Record longer samples (30+ seconds)
4. Fine-tune on target voice dataset
5. Verify correct reference tokens are being used

#### Poor Audio Quality

**Symptoms:** Crackling, artifacts, low fidelity

**Solutions:**
1. Check input audio quality and preprocessing
2. Verify correct model checkpoints are loaded
3. Ensure consistent sampling rates
4. Apply loudness normalization to dataset
5. Move to quieter, less echoey recording space
6. Try different recording device

#### Model Won't Load

**Symptoms:** Errors loading checkpoints, missing files

**Solutions:**
1. Verify checkpoint paths are correct
2. Re-download model weights:
   ```bash
   huggingface-cli download fishaudio/openaudio-s1-mini --local-dir checkpoints/openaudio-s1-mini
   ```
3. Check all dependencies are installed
4. Ensure sufficient disk space

#### GPU/Memory Errors

**Symptoms:** CUDA out of memory, crashes during training/inference

**Solutions:**
1. Reduce batch size in config
2. Use bf16/mixed precision training
3. Use LoRA instead of full fine-tuning
4. Add `--half` parameter for GPUs without bf16 support
5. Enable gradient checkpointing

#### No Sound Output

**Symptoms:** Synthesis completes but audio file is silent or corrupted

**Solutions:**
1. Check device audio settings
2. Verify output file integrity (play with external player)
3. Ensure semantic tokens generated correctly
4. Check vocoder checkpoint is correct
5. Verify file format compatibility

#### Voice Cloning Fails

**Symptoms:** Clone doesn't sound like target, or fails to generate

**Solutions:**
1. Re-extract codes from clean reference sample
2. Ensure reference audio is high quality (no background noise)
3. Use reference text that matches audio content
4. Try different prompt conditioning parameters
5. Verify reference audio length (10-30 seconds)

### Performance Optimization Tips

1. **Enable Compilation:**
   ```bash
   # Add --compile flag for ~10x speedup
   python fish_speech/models/text2semantic/inference.py --compile
   ```

2. **Batch Processing:**
   - Process multiple texts in batches
   - Reuse loaded models across requests

3. **Hardware Recommendations:**
   - Minimum: 12GB VRAM for smooth inference
   - Recommended: RTX 4090 or better for best performance
   - CPU inference possible but much slower

---

## Advanced Tips and Tricks

### Phoneme Control

For precise pronunciation of tricky words, names, or technical terms:

```
<|phoneme_start|>word|phoneme_end|>
```

**Use cases:**
- Foreign names
- Technical jargon
- Code-switching scenarios
- Disambiguation of homophones

### Paralinguistic Effects

Humanize speech with non-verbal vocalizations:

```
(laughing) That's so funny!
Let me think... (sighing) I suppose you're right.
(breathing) We made it! (panting)
```

### Batch Voice Creation

For projects with multiple characters or voices:

1. Create separate reference folders for each voice:
   ```
   references/
   ├── character1/
   │   ├── sample.lab
   │   └── sample.wav
   ├── character2/
   │   ├── sample.lab
   │   └── sample.wav
   ```

2. Access quickly in WebUI or API by reference_id

### Emotional Variation in Long-Form Content

For audiobooks, podcasts, or narrative content:

1. **Mix emotions naturally:**
   ```
   (calm) The story begins on a quiet morning. 
   (excited) Suddenly, something incredible happened! 
   (worried) But danger was approaching.
   ```

2. **Use subtle markers:**
   - (soft tone) for intimate moments
   - (whispering) for secrets
   - (in a hurry tone) for action sequences

### Dataset Mixing Strategies

**For Multi-Speaker Models:**
- Balance speaker distribution (avoid one speaker dominating)
- Ensure each speaker has varied emotional samples
- Mix languages if building multilingual model

**For Emotional Enhancement:**
- Start with 70% target speaker, 30% external emotional data
- Gradually adjust ratio based on results
- Monitor speaker identity preservation

### Testing Workflows

**Automated Testing:**
```bash
# Create test scripts for batch synthesis
for text_file in test_cases/*.txt; do
    python fish_speech/models/text2semantic/inference.py \
        --text "$(cat $text_file)" \
        --prompt-tokens "reference.npy" \
        --compile
done
```

**Quality Metrics:**
- Word Error Rate (WER) - accuracy
- Character Error Rate (CER) - precision
- Speaker Distance - similarity to reference
- Mean Opinion Score (MOS) - human evaluation

### Community Resources

- **GitHub Discussions:** [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech/discussions)
- **Discord:** [Join Fish Audio Discord](https://discord.gg/Es5qTB9BcN)
- **Documentation:** [docs.fish.audio](https://docs.fish.audio/)
- **API Playground:** [fishspeech.net/en/api-playground](https://fishspeech.net/en/api-playground)

### Gradio Environment Variables

Customize WebUI deployment:

```bash
export GRADIO_SHARE=1                    # Enable public sharing
export GRADIO_SERVER_PORT=7860           # Change port
export GRADIO_SERVER_NAME=0.0.0.0        # Allow external access

python -m tools.run_webui
```

---

## Summary: Quick Reference Checklist

### For Best Voice Cloning Results:
- [ ] Record in quiet environment (bedroom, parked car)
- [ ] Use decent microphone (USB mic or quality headset)
- [ ] Record 20-30 seconds of natural speech
- [ ] Maintain consistent tone and volume
- [ ] Only one speaker per recording
- [ ] Apply loudness normalization
- [ ] Test with various text samples

### For Training/Fine-Tuning:
- [ ] Organize dataset with proper structure
- [ ] Include emotional variety if needed
- [ ] Extract semantic tokens
- [ ] Configure batch size for your GPU
- [ ] Start with 10-20 epochs
- [ ] Monitor for overfitting
- [ ] Use earliest checkpoint that works

### For Consistent Results:
- [ ] Use reference audio effectively
- [ ] Leverage prompt engineering
- [ ] Save reference profiles
- [ ] Use emotion markers appropriately
- [ ] Fine-tune on target voice if needed
- [ ] Test across different scenarios

### For Best Performance:
- [ ] Enable --compile for inference speedup
- [ ] Use 12GB+ VRAM GPU
- [ ] Batch process when possible
- [ ] Use LoRA for memory efficiency
- [ ] Monitor resource usage

---

## References and Further Reading

1. **Official Documentation:** [docs.fish.audio](https://docs.fish.audio/)
2. **GitHub Repository:** [github.com/fishaudio/fish-speech](https://github.com/fishaudio/fish-speech)
3. **Research Paper:** [arXiv:2411.01156](https://arxiv.org/abs/2411.01156)
4. **Voice Cloning Best Practices:** [docs.fish.audio/developer-guide/best-practices/voice-cloning](https://docs.fish.audio/developer-guide/best-practices/voice-cloning)
5. **Emotion Control Documentation:** [docs.fish.audio/developer-guide/core-features/emotions](https://docs.fish.audio/developer-guide/core-features/emotions)
6. **API Documentation:** [fishspeech.net/en/api-playground](https://fishspeech.net/en/api-playground)
7. **Fish Audio Platform:** [fish.audio](https://fish.audio/)

---

**Document Version:** 1.0  
**Last Updated:** January 2026  
**Contributors:** Research compiled from official Fish Audio documentation, community best practices, and technical papers

For questions or contributions to this guide, please open an issue on the [Fish Speech GitHub repository](https://github.com/fishaudio/fish-speech/issues).
