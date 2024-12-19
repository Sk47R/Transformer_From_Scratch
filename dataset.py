import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Any

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index: Any) -> Any:
        src_target_pair = self.ds[index] # ORIGINAL PAIR FROM HUGGING FACE
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair["translation"][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids # GIVES INPUT IDS CORRESPONDING TO EACH WORD IN THE INPUT SENTENCE. GIVES IN THE FORM OF ARRAY
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # MINUS 2 AS WE ADD SOS AND EOS TOKEN
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # IN TRAINING WE ONLY ADD SOS IN DECODER AND IN LABEL ONLY EOS TOKEN.

        # ENSURE seq_len IS ENOUGH TO REPRESENT ALL SENTENCES IN THE DATASET, AND THROUGH ERROR IF SENTENCE > seq_len
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")
        
        # BUILD THE TENSORS FOR ENCODER INPUT, DECODER INPUT, AND LABEL (OUTPUT OF DECODER)
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype = torch.int64)
            ]
        )           # ADDIND SOS AND EOS TO SOURCE TEXT

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64)
            ]
        )           # ADD ONLY <SOS>

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype = torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype = torch.int64)
            ]
        )           # ADD ONLY <EOS>

        # Double check the size of the tensors to make sure they are all seq_len long
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, # (seq_len)
            "decoder_input": decoder_input, # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (1, seq_len, seq_len)
            "label": label, # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,

        }
    

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal = 1).type(torch.int) # RETURN ALL VALUE ABOVE DIAGONAL 
    return mask == 0       # RETURN ALL VALUES BELOW THE DIAGONAL