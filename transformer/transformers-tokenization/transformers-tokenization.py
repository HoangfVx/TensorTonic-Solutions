from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """

    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0

        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        All text is lowercased.
        """
        # Fixed IDs for special tokens
        self.word_to_id = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }

        # Collect unique words
        words = set()
        for text in texts:
            words.update(text.lower().split())

        # Add words in sorted order
        next_id = 4
        for word in sorted(words):
            self.word_to_id[word] = next_id
            next_id += 1

        # Build reverse mapping
        self.id_to_word = {
            idx: word for word, idx in self.word_to_id.items()
        }

        self.vocab_size = len(self.word_to_id)

    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        tokens = text.lower().split()
    
        return [
            self.word_to_id.get(
                token,
                self.word_to_id[self.unk_token]
            )
            for token in tokens
        ]

    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        Skip PAD, BOS, EOS if present.
        """
        special_tokens = {
            self.pad_token,
            self.bos_token,
            self.eos_token,
        }
    
        words = []
    
        for idx in ids:
            word = self.id_to_word.get(idx, self.unk_token)
    
            if word in special_tokens:
                continue
    
            words.append(word)
    
        return " ".join(words)