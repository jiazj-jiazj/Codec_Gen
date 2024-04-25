from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from valle.utils import SymbolTable


class TextTokenCollater:
    """Collate list of text tokens

    Map sentences to integers. Sentences are padded to equal length.
    Beginning and end-of-sequence symbols can be added.

    Example:
        >>> token_collater = TextTokenCollater(text_tokens)
        >>> tokens_batch, tokens_lens = token_collater(text)

    Returns:
        tokens_batch: IntTensor of shape (B, L)
            B: batch dimension, number of input sentences
            L: length of the longest sentence
        tokens_lens: IntTensor of shape (B,)
            Length of each sentence after adding <eos> and <bos>
            but before padding.
    """

    def __init__(
        self,
        text_tokens: List[str],
        add_eos: bool = True,
        add_bos: bool = True,
        pad_symbol: str = "<pad>",
        bos_symbol: str = "<bos>",
        eos_symbol: str = "<eos>",
    ):
        self.pad_symbol = pad_symbol

        self.add_eos = add_eos
        self.add_bos = add_bos

        self.bos_symbol = bos_symbol
        self.eos_symbol = eos_symbol

        unique_tokens = (
            [pad_symbol]
            + ([bos_symbol] if add_bos else [])
            + ([eos_symbol] if add_eos else [])
            + sorted(text_tokens)
        )
        self.token2idx = {token: idx for idx, token in enumerate(unique_tokens)}
        self.idx2token = [token for token in unique_tokens]

    def index( 
        self, tokens_list: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seqs, seq_lens = [], []
        for tokens in tokens_list:
            assert (
                all([True if s in self.token2idx else False for s in tokens])
                is True
            )
            seq = (
                ([self.bos_symbol] if self.add_bos else [])
                + list(tokens)
                + ([self.eos_symbol] if self.add_eos else [])
            )
            seqs.append(seq)
            seq_lens.append(len(seq))

        max_len = max(seq_lens)
        for k, (seq, seq_len) in enumerate(zip(seqs, seq_lens)):
            seq.extend([self.pad_symbol] * (max_len - seq_len))

        tokens = torch.from_numpy(
            np.array(
                [[self.token2idx[token] for token in seq] for seq in seqs],
                dtype=np.int64,
            )
        )
        tokens_lens = torch.IntTensor(seq_lens)

        return tokens, tokens_lens

    def __call__(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:

        if isinstance(texts[0][0], list): #[a,b]
            embedding_size=2 
        else:
            embedding_size=1 # a
        # print(f"embedding_size is {embedding_size}")

        tokens_seqs = [[p for p in text] for text in texts]
        max_len = len(max(tokens_seqs, key=len))

        if embedding_size==1:
            seqs = [
                ([self.bos_symbol] if self.add_bos else [])
                + list(seq)
                + ([self.eos_symbol] if self.add_eos else [])
                + [self.pad_symbol] * (max_len - len(seq))
                for seq in tokens_seqs
            ]
            tokens_batch = torch.from_numpy(
                np.array(
                    [[self.token2idx[str(token)] for token in seq] for seq in seqs],
                    dtype=np.int64,
                )
            )
        elif embedding_size==2:
            seqs = [
                ([[self.bos_symbol, self.bos_symbol]] if self.add_bos else [])
                + list(seq)
                + ([[self.eos_symbol, self.eos_symbol]] if self.add_eos else [])
                + [[self.pad_symbol, self.pad_symbol]] * (max_len - len(seq))
                for seq in tokens_seqs
            ]
            tokens_batch = torch.from_numpy(
                np.array(
                    [[[ self.token2idx[str(token_part)] for token_part in token] for token in seq] for seq in seqs],
                    dtype=np.int64,
                )
            )

        tokens_lens = torch.IntTensor(
            [
                len(seq) + int(self.add_eos) + int(self.add_bos)
                for seq in tokens_seqs
            ]
        )
        return tokens_batch, tokens_lens


def get_text_token_collater(text_tokens_file: str) -> TextTokenCollater:
    text_tokens_path = Path(text_tokens_file)
    unique_tokens = SymbolTable.from_file(text_tokens_path)
    collater = TextTokenCollater(
        unique_tokens.symbols, add_bos=True, add_eos=True
    )
    return collater
