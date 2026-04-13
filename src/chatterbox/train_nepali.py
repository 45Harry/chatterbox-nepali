
import argparse
import os
import json
from pathlib import Path
import torch.multiprocessing as mp

# Fix librosa multiprocessing issue
mp.set_start_method('spawn', force=True)

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import librosa
import numpy as np
from safetensors.torch import save_file, load_file as load_safetensors

from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals
from chatterbox.models.t3.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.s3tokenizer import S3_SR, S3Tokenizer
from chatterbox.models.s3gen import S3Gen
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.models.tokenizers import MTLTokenizer

class NepaliDataset(Dataset):
    def __init__(self, manifest_path, tokenizer, s3_tokenizer, voice_encoder, device):
        self.tokenizer = tokenizer
        self.s3_tokenizer = s3_tokenizer
        self.voice_encoder = voice_encoder
        self.device = device
        self.data = []

        manifest_path = Path(manifest_path)
        if manifest_path.suffix == '.jsonl':
            with open(manifest_path, 'r', encoding='utf-8') as f:
                self.data = [json.loads(line) for line in f]
        else:
            # Handle CSV (usually | for this dataset) or TSV (\t)
            delimiter = '\t' if manifest_path.suffix == '.tsv' else '|'
            wav_dir = manifest_path.parent / "wavs"
            
            with open(manifest_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(delimiter)
                    if len(parts) >= 2:
                        fname = parts[0]
                        text = parts[1]
                        # Support potential [ne] tag in text or just raw text
                        audio_path = wav_dir / f"{fname}.wav"
                        if audio_path.exists():
                            self.data.append({"audio_path": str(audio_path), "text": text})
            
            if not self.data:
                print(f"⚠️ Warning: No data loaded from {manifest_path}. Check delimiters or wav path.")
            else:
                print(f"✅ Loaded {len(self.data)} items from {manifest_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = item.get('audio_path') or item.get('audio')
        text = item['text']

        # Do NOT prepend [START] in text since MTLTokenizer adds [ne] as prefix to the text.
        # This will misorder tokens as [ne] [START] instead of inference which uses [START] [ne].
        # Better to pad the tokens manually.
        text_tokens = self.tokenizer.text_to_tokens(text, language_id='ne', lowercase=False).squeeze(0)
        
        # T3 expects [START] and [STOP] tokens (IDs 255 and 0)
        # Pad them manually exactly as inference `mtl_tts.py -> generate()` does
        text_tokens = F.pad(text_tokens, (1, 0), value=255) # start_text_token
        text_tokens = F.pad(text_tokens, (0, 1), value=0)   # stop_text_token

        # Load audio
        wav, _ = librosa.load(audio_path, sr=S3_SR)
        
        # Audio to speech tokens
        with torch.no_grad():
            speech_tokens, _ = self.s3_tokenizer.forward([wav])
            speech_tokens = speech_tokens.squeeze(0)
            
            # Speaker embedding
            ve_embed = torch.from_numpy(self.voice_encoder.embeds_from_wavs([wav], sample_rate=S3_SR))
            ve_embed = ve_embed.mean(axis=0, keepdim=True)

        return {
            "text_tokens": text_tokens,
            "speech_tokens": speech_tokens,
            "speaker_emb": ve_embed,
            "wav": wav
        }

def collate_fn(batch):
    text_tokens = [item['text_tokens'] for item in batch]
    speech_tokens = [item['speech_tokens'] for item in batch]
    speaker_embs = [item['speaker_emb'] for item in batch]

    text_token_lens = torch.tensor([len(t) for t in text_tokens], dtype=torch.long)
    speech_token_lens = torch.tensor([len(s) for s in speech_tokens], dtype=torch.long)

    # Pad sequences
    text_tokens_padded = torch.nn.utils.rnn.pad_sequence(text_tokens, batch_first=True, padding_value=0).long()
    speech_tokens_padded = torch.nn.utils.rnn.pad_sequence(speech_tokens, batch_first=True, padding_value=6562).long() # stop_speech_token
    
    speaker_embs = torch.cat(speaker_embs, dim=0)
    
    return {
        "text_tokens": text_tokens_padded,
        "text_token_lens": text_token_lens,
        "speech_tokens": speech_tokens_padded,
        "speech_token_lens": speech_token_lens,
        "speaker_emb": speaker_embs
    }

def train(args):
    device = torch.device(args.device)

    # TensorBoard writer
    tb_log_dir = f"runs/nepali_tts_{Path(args.manifest).stem}"
    tb_writer = SummaryWriter(tb_log_dir)
    print(f"📊 TensorBoard logs at: {tb_log_dir}")
    print(f"   Run: tensorboard --logdir=runs")
    
    # Load pretrained components
    if args.ckpt_dir:
        model_wrapper = ChatterboxMultilingualTTS.from_local(args.ckpt_dir, device)
    else:
        model_wrapper = ChatterboxMultilingualTTS.from_pretrained(device)
    
    t3 = model_wrapper.t3
    tokenizer = model_wrapper.tokenizer
    
    # Optional: Resume from an intermediate training checkpoint
    if args.resume_t3_weights:
        print(f"🔄 Resuming training from {args.resume_t3_weights}...")
        resume_state = torch.load(args.resume_t3_weights, map_location="cpu", weights_only=True)
        
        # Clean state dict keys (strip 'patched_model.' and 'model.' if they exist)
        cleaned_state = {k.replace("patched_model.", "").replace("model.", ""): v for k, v in resume_state.items()}
        
        # Handle vocabulary size potentially being different (e.g. if you added [ne])
        current_vocab_size = t3.hp.text_tokens_dict_size
        ckpt_vocab_size = cleaned_state["text_emb.weight"].shape[0]
        if current_vocab_size != ckpt_vocab_size:
            t3.resize_text_embeddings(ckpt_vocab_size)
            t3.load_state_dict(cleaned_state, strict=False)
            t3.resize_text_embeddings(current_vocab_size)
        else:
            t3.load_state_dict(cleaned_state, strict=False)
        t3.to(device)
    
    # For data loading, we keep feature extractors strictly on CPU
    s3_tokenizer = model_wrapper.s3gen.tokenizer.cpu()
    voice_encoder = model_wrapper.ve.cpu()

    # OPTIMIZATION: Initialize [ne] (Nepali) tag from [hi] (Hindi) tag
    # This prevents the initial gibberish by starting with a related language!
    # Only do this when starting fresh (not resuming), since resumed weights are already tuned.
    hi_idx = 722
    ne_idx = 2454
    if not args.resume_t3_weights:
        with torch.no_grad():
            if t3.text_emb.weight.shape[0] > ne_idx:
                print(f"🎯 Initializing [ne] tag ({ne_idx}) weights from [hi] tag ({hi_idx})...")
                # 1. Update Embedding
                t3.text_emb.weight[ne_idx] = t3.text_emb.weight[hi_idx].clone()
                # 2. Update Prediction Head
                t3.text_head.weight[ne_idx] = t3.text_head.weight[hi_idx].clone()
    
    t3.train()
    
    # Dataset
    dataset = NepaliDataset(args.manifest, tokenizer, s3_tokenizer, voice_encoder, device="cpu")
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if args.device != "cpu" else False
    )
    
    optimizer = AdamW(t3.parameters(), lr=args.lr)
    
    start_epoch = 0
    if args.resume_t3_weights:
        import re
        match = re.search(r"epoch_(\d+)", args.resume_t3_weights)
        if match:
            start_epoch = int(match.group(1)) + 1
            print(f"⏩ Setting internal loop iterator to start at Epoch {start_epoch}")
            
    for epoch in range(start_epoch, args.epochs):
        losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(pbar):
            
            text_tokens = batch['text_tokens'].to(device)
            text_token_lens = batch['text_token_lens'].to(device)
            speech_tokens = batch['speech_tokens'].to(device)
            speech_token_lens = batch['speech_token_lens'].to(device)
            speaker_emb = batch['speaker_emb'].to(device)
            
            # Prepare T3Cond
            prompt_len = t3.hp.speech_cond_prompt_len
            cond_prompt_tokens = speech_tokens[:, :prompt_len] if speech_tokens.size(1) > prompt_len else None
            
            t3_cond = T3Cond(
                speaker_emb=speaker_emb,
                cond_prompt_speech_tokens=cond_prompt_tokens,
                emotion_adv=0.5 * torch.ones(text_tokens.size(0), 1, 1, device=device)
            )
            
            loss_text, loss_speech = t3.loss(
                t3_cond=t3_cond,
                text_tokens=text_tokens,
                text_token_lens=text_token_lens,
                speech_tokens=speech_tokens,
                speech_token_lens=speech_token_lens
            )
            
            loss = (loss_text + loss_speech) / args.accum_steps
            loss.backward()
            
            if (i + 1) % args.accum_steps == 0 or (i + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            pbar.set_postfix({"loss": loss.item() * args.accum_steps, "loss_speech": loss_speech.item()})

            # Track for epoch summary
            losses.append(loss.item() * args.accum_steps)

            # TensorBoard logging (per-step)
            global_step = epoch * len(dataloader) + i
            tb_writer.add_scalar("train/loss", loss.item() * args.accum_steps, global_step)
            tb_writer.add_scalar("train/loss_text", loss_text.item(), global_step)
            tb_writer.add_scalar("train/loss_speech", loss_speech.item(), global_step)
            
            # Memory Fixes for M2 Max
            del loss, loss_text, loss_speech, t3_cond
            if i % 10 == 0 and device.type == "mps":
                torch.mps.empty_cache()
            
        # Epoch summary
        avg_loss = sum(losses) / len(losses) if losses else float('inf')
        tb_writer.add_scalar("epoch/avg_loss", avg_loss, epoch)
        print(f"📈 Epoch {epoch} complete — avg loss: {avg_loss:.4f}")

        # Save checkpoint (skip validation sample generation to avoid CUDA crashes)
        if epoch % args.save_every == 0:
            os.makedirs("results", exist_ok=True)
            ckpt_path = f"results/t3_nepali_epoch_{epoch}.pt"
            torch.save(t3.state_dict(), ckpt_path)
            print(f"✅ Checkpoint saved: {ckpt_path}")
    # Safetensors errors out if shared memory pointers exist.
    # Because 'patched_model' is a duplicated reference specifically made for the generate validation loop, it crashes. We strip it!
    final_sd = {k: v for k, v in t3.state_dict().items() if not k.startswith("patched_model.")}
    os.makedirs("results", exist_ok=True)
    save_file(final_sd, "results/t3_mtl_nepali_final.safetensors")
    print("Training finished. Saved to results/t3_mtl_nepali_final.safetensors")

    tb_writer.close()

if __name__ == "__main__":
    def get_default_device():
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest (jsonl, csv, or tsv)")
    parser.add_argument("--ckpt_dir", type=str, help="Path to base pretrained model dir")
    parser.add_argument("--resume_t3_weights", type=str, help="Path to a t3_nepali_epoch_X.pt file to resume from")
    parser.add_argument("--device", type=str, default=get_default_device(), help="Device to use (cpu, cuda, mps)")
    parser.add_argument("--eval_audio_path", type=str, help="Path to reference audio for validation samples")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (M2 Max can handle 16-32)")
    parser.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps to simulate larger batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of CPU threads for background data loading")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for fine-tuning")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate (usually 1e-5 to 5e-5)")
    parser.add_argument("--save_every", type=int, default=5, help="Save interval in epochs")
    
    args = parser.parse_args()
    
    if args.device == "mps":
        print("🚀 Using MPS (Metal Performance Shaders) for Mac acceleration")
    
    train(args)
