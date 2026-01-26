"""
Solution 3: Hybrid CNN-LSTM with Multi-Scale Attention for Hardware Trojan Detection
=====================================================================================

This solution combines:
1. Multi-scale 1D CNNs for local pattern extraction (detecting trojan triggers)
2. Bidirectional LSTMs for sequential dependency modeling
3. Self-attention mechanism for long-range dependencies
4. Feature fusion with learned importance weights

Key Innovations:
- Multi-scale convolutions capture patterns of different sizes (operators, statements, blocks)
- Temporal modeling via BiLSTM captures sequential code structure
- Attention highlights suspicious code regions
- Ensemble of structural and semantic features

Author: Hardware Security Research
Date: 2025
"""

import os
import re
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

class VerilogFeatureExtractor:
    """
    Multi-modal feature extractor for Verilog code

    Extracts:
    1. Token-level features (vocabulary-based)
    2. Structural features (nesting, blocks, signals)
    3. Statistical features (complexity metrics)
    4. Sequence features (n-grams, patterns)
    """

    def __init__(self, vocab_size: int = 5000, max_seq_len: int = 4096):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Vocabulary
        self.token_to_id = {'<PAD>': 0, '<UNK>': 1}
        self.id_to_token = {0: '<PAD>', 1: '<UNK>'}

        # Verilog keywords for classification
        self.keywords = {
            'module', 'endmodule', 'input', 'output', 'inout', 'wire', 'reg',
            'assign', 'always', 'initial', 'begin', 'end', 'if', 'else',
            'case', 'casex', 'casez', 'endcase', 'for', 'while', 'posedge',
            'negedge', 'parameter', 'localparam', 'integer', 'real', 'function',
            'task', 'generate', 'endgenerate', 'default'
        }

        # Operators
        self.operators = set('+-*/%&|^~<>=!?:')

        # Add keywords to vocabulary
        for kw in sorted(self.keywords):
            self._add_token(kw)

        # Structural feature names
        self.structural_features = [
            'num_modules', 'num_inputs', 'num_outputs', 'num_wires', 'num_regs',
            'num_always', 'num_assign', 'num_if', 'num_case', 'num_for',
            'max_nesting_depth', 'avg_nesting_depth', 'num_operators',
            'num_conditionals', 'code_lines', 'comment_ratio',
            'signal_count', 'unique_signals', 'operator_density',
            'conditional_density', 'sequential_blocks', 'combinational_blocks'
        ]

    def _add_token(self, token: str) -> int:
        """Add token to vocabulary"""
        if token not in self.token_to_id and len(self.token_to_id) < self.vocab_size:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            return idx
        return self.token_to_id.get(token, 1)

    def tokenize(self, code: str) -> List[str]:
        """Tokenize Verilog code"""
        # Remove comments
        code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

        # Tokenize
        tokens = re.findall(r'[a-zA-Z_]\w*|\d+\'[bBdDhHoO][\da-fA-F_xXzZ]+|\d+|[^\s\w]', code)
        return tokens

    def extract_sequence_features(self, code: str) -> np.ndarray:
        """Extract sequence of token IDs"""
        tokens = self.tokenize(code)

        # Convert to IDs
        ids = []
        for token in tokens[:self.max_seq_len]:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            elif token in self.keywords:
                ids.append(self._add_token(token))
            elif len(self.token_to_id) < self.vocab_size:
                ids.append(self._add_token(token))
            else:
                ids.append(1)  # UNK

        # Pad sequence
        if len(ids) < self.max_seq_len:
            ids.extend([0] * (self.max_seq_len - len(ids)))

        return np.array(ids[:self.max_seq_len], dtype=np.int64)

    def extract_structural_features(self, code: str) -> np.ndarray:
        """Extract structural/statistical features"""
        features = {}

        # Module counts
        features['num_modules'] = len(re.findall(r'\bmodule\b', code))
        features['num_inputs'] = len(re.findall(r'\binput\b', code))
        features['num_outputs'] = len(re.findall(r'\boutput\b', code))
        features['num_wires'] = len(re.findall(r'\bwire\b', code))
        features['num_regs'] = len(re.findall(r'\breg\b', code))

        # Block counts
        features['num_always'] = len(re.findall(r'\balways\b', code))
        features['num_assign'] = len(re.findall(r'\bassign\b', code))
        features['num_if'] = len(re.findall(r'\bif\b', code))
        features['num_case'] = len(re.findall(r'\bcase[xz]?\b', code))
        features['num_for'] = len(re.findall(r'\bfor\b', code))

        # Nesting depth
        depths = self._compute_nesting_depths(code)
        features['max_nesting_depth'] = max(depths) if depths else 0
        features['avg_nesting_depth'] = np.mean(depths) if depths else 0

        # Operators
        features['num_operators'] = sum(1 for c in code if c in self.operators)
        features['num_conditionals'] = code.count('?')

        # Lines and comments
        lines = code.split('\n')
        features['code_lines'] = len(lines)
        comment_lines = sum(1 for l in lines if l.strip().startswith('//'))
        features['comment_ratio'] = comment_lines / max(len(lines), 1)

        # Signal analysis
        signals = re.findall(r'\b([a-zA-Z_]\w*)\b', code)
        features['signal_count'] = len(signals)
        features['unique_signals'] = len(set(signals))

        # Density metrics
        features['operator_density'] = features['num_operators'] / max(len(code), 1) * 100
        features['conditional_density'] = features['num_conditionals'] / max(features['code_lines'], 1)

        # Block type analysis
        sequential_patterns = re.findall(r'always\s*@\s*\(\s*(?:pos|neg)edge', code)
        features['sequential_blocks'] = len(sequential_patterns)
        features['combinational_blocks'] = features['num_always'] - features['sequential_blocks']

        # Return as array in consistent order
        return np.array([features[name] for name in self.structural_features], dtype=np.float32)

    def _compute_nesting_depths(self, code: str) -> List[int]:
        """Compute nesting depth throughout the code"""
        depths = []
        current_depth = 0

        for char in code:
            if char in '({[':
                current_depth += 1
            elif char in ')}]':
                current_depth = max(0, current_depth - 1)
            depths.append(current_depth)

        return depths

    def extract_all_features(self, code: str) -> Dict[str, np.ndarray]:
        """Extract all feature types"""
        return {
            'sequence': self.extract_sequence_features(code),
            'structural': self.extract_structural_features(code),
            'length': np.array([len(self.tokenize(code))], dtype=np.int64)
        }

    def build_vocab(self, files: List[str]):
        """Build vocabulary from corpus"""
        counter = Counter()

        for filepath in files:
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                tokens = self.tokenize(code)
                counter.update(tokens)
            except Exception as e:
                continue

        # Add most common tokens
        for token, _ in counter.most_common(self.vocab_size - len(self.token_to_id)):
            self._add_token(token)

        print(f"Vocabulary size: {len(self.token_to_id)}")


# ============================================================================
# NEURAL NETWORK COMPONENTS
# ============================================================================

class MultiScaleCNN(nn.Module):
    """
    Multi-scale 1D CNN for capturing patterns of different sizes

    Uses parallel convolutions with different kernel sizes to capture:
    - Small patterns (3): operators, small expressions
    - Medium patterns (5-7): statements, conditions
    - Large patterns (11-15): code blocks, functions
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 kernel_sizes: List[int] = [3, 5, 7, 11, 15],
                 dropout: float = 0.2):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim // len(kernel_sizes),
                         kernel_size=k, padding=k//2),
                nn.BatchNorm1d(hidden_dim // len(kernel_sizes)),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for k in kernel_sizes
        ])

        # Second layer convolutions
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, seq_len, hidden_dim)
        """
        # Transpose for conv1d: (batch, input_dim, seq_len)
        x = x.transpose(1, 2)

        # Multi-scale convolutions
        conv_outputs = [conv(x) for conv in self.convs]

        # Concatenate along channel dimension
        x = torch.cat(conv_outputs, dim=1)

        # Second layer
        x = self.conv2(x)

        # Transpose back: (batch, seq_len, hidden_dim)
        return x.transpose(1, 2)


class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM for sequential modeling
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,  # Bidirectional doubles this
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.output_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
            lengths: actual sequence lengths for packing
        Returns:
            (batch, seq_len, hidden_dim)
        """
        if lengths is not None:
            # Pack sequences
            lengths = lengths.cpu().clamp(min=1)
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            output, _ = self.lstm(packed)
            output, _ = pad_packed_sequence(output, batch_first=True, total_length=x.size(1))
        else:
            output, _ = self.lstm(x)

        return self.dropout(output)


class SelfAttention(nn.Module):
    """
    Self-attention mechanism for capturing long-range dependencies
    """

    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            mask: attention mask
        Returns:
            output: (batch, seq_len, hidden_dim)
            attention_weights: (batch, seq_len, seq_len)
        """
        # Self-attention with residual
        attn_out, attn_weights = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm(x + self.dropout(attn_out))

        return x, attn_weights


class AttentionPooling(nn.Module):
    """
    Attention-based pooling for sequence aggregation
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) - True for valid positions
        Returns:
            pooled: (batch, hidden_dim)
            attention_weights: (batch, seq_len)
        """
        # Compute attention scores
        scores = self.attention(x).squeeze(-1)  # (batch, seq_len)

        # Mask invalid positions
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        # Softmax
        weights = F.softmax(scores, dim=-1)

        # Weighted sum
        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)

        return pooled, weights


# ============================================================================
# MAIN MODEL
# ============================================================================

class HybridCNNLSTMTrojanDetector(nn.Module):
    """
    Hybrid CNN-LSTM Model for Hardware Trojan Detection

    Architecture:
    1. Embedding layer for token sequences
    2. Multi-scale CNN for local pattern extraction
    3. BiLSTM for sequential modeling
    4. Self-attention for long-range dependencies
    5. Feature fusion with structural features
    6. Classification head

    Features:
    - Multi-scale convolutions capture trojan triggers of various sizes
    - BiLSTM models sequential code structure
    - Attention highlights suspicious regions
    - Structural features provide global context
    """

    def __init__(self,
                 vocab_size: int = 5000,
                 embed_dim: int = 128,
                 hidden_dim: int = 256,
                 num_structural_features: int = 22,
                 num_lstm_layers: int = 2,
                 num_attention_heads: int = 8,
                 kernel_sizes: List[int] = [3, 5, 7, 11, 15],
                 num_classes: int = 2,
                 dropout: float = 0.3):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Multi-scale CNN
        self.cnn = MultiScaleCNN(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            kernel_sizes=kernel_sizes,
            dropout=dropout
        )

        # BiLSTM
        self.lstm = BiLSTMEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_lstm_layers,
            dropout=dropout
        )

        # Self-attention
        self.self_attention = SelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )

        # Attention pooling
        self.attention_pool = AttentionPooling(hidden_dim)

        # Structural feature encoder
        self.structural_encoder = nn.Sequential(
            nn.Linear(num_structural_features, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU()
        )

        # Feature fusion
        # Combine: attention_pooled + max_pooled + mean_pooled + structural
        fusion_dim = hidden_dim * 3 + hidden_dim // 2

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Token-level anomaly detection head
        self.token_anomaly = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(self,
                sequence: torch.Tensor,
                structural: torch.Tensor,
                lengths: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass

        Args:
            sequence: Token IDs (batch, seq_len)
            structural: Structural features (batch, num_features)
            lengths: Actual sequence lengths
            return_attention: Whether to return attention weights

        Returns:
            logits: Classification logits (batch, num_classes)
            token_scores: Per-token anomaly scores (optional)
            attention_weights: Attention weights (optional)
        """
        batch_size, seq_len = sequence.shape

        # Create mask
        mask = (sequence != 0)  # (batch, seq_len)

        # Embedding
        x = self.embedding(sequence)  # (batch, seq_len, embed_dim)

        # Multi-scale CNN
        x = self.cnn(x)  # (batch, seq_len, hidden_dim)

        # BiLSTM
        x = self.lstm(x, lengths)  # (batch, seq_len, hidden_dim)

        # Self-attention
        x, self_attn_weights = self.self_attention(x, ~mask)

        # Multiple pooling strategies
        # 1. Attention pooling
        attn_pooled, pool_weights = self.attention_pool(x, mask)

        # 2. Max pooling
        x_masked = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        max_pooled = x_masked.max(dim=1)[0]

        # 3. Mean pooling
        x_masked_zero = x * mask.unsqueeze(-1).float()
        mean_pooled = x_masked_zero.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

        # Encode structural features
        struct_encoded = self.structural_encoder(structural)

        # Fusion
        fused = torch.cat([attn_pooled, max_pooled, mean_pooled, struct_encoded], dim=-1)
        fused = self.fusion(fused)

        # Classification
        logits = self.classifier(fused)

        if return_attention:
            # Token-level anomaly scores
            token_scores = self.token_anomaly(x).squeeze(-1)
            return logits, token_scores, pool_weights

        return logits


# ============================================================================
# DATASET
# ============================================================================

class HybridTrojanDataset(Dataset):
    """
    Dataset for Hybrid CNN-LSTM model
    """

    def __init__(self, data_dir: str, feature_extractor: VerilogFeatureExtractor,
                 max_seq_len: int = 4096):
        self.data_dir = data_dir
        self.feature_extractor = feature_extractor
        self.max_seq_len = max_seq_len

        self.samples = []
        self.labels = []
        self.file_paths = []

        self._load_dataset()

    def _load_dataset(self):
        """Load all files"""
        label_map = {
            'TjIn': 1,
            'TjFree': 0,
            'IP-RTL-toy': 0,
            'IP-Netlist-toy': 0
        }

        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.v'):
                    filepath = os.path.join(root, file)

                    label = 0
                    for key, val in label_map.items():
                        if key in root:
                            label = val
                            break

                    self.file_paths.append(filepath)
                    self.labels.append(label)

        print(f"Loaded {len(self.file_paths)} files")
        print(f"  Clean: {self.labels.count(0)}, Trojaned: {self.labels.count(1)}")

        # Build vocabulary
        self.feature_extractor.build_vocab(self.file_paths)

        # Pre-extract features
        self._preprocess()

    def _preprocess(self):
        """Pre-extract all features"""
        structural_features = []

        for filepath in self.file_paths:
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    code = f.read()
                features = self.feature_extractor.extract_all_features(code)
                self.samples.append(features)
                structural_features.append(features['structural'])
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                # Dummy features
                dummy = {
                    'sequence': np.zeros(self.max_seq_len, dtype=np.int64),
                    'structural': np.zeros(len(self.feature_extractor.structural_features), dtype=np.float32),
                    'length': np.array([1], dtype=np.int64)
                }
                self.samples.append(dummy)
                structural_features.append(dummy['structural'])

        # Normalize structural features
        structural_array = np.array(structural_features)
        self.scaler = StandardScaler()
        normalized = self.scaler.fit_transform(structural_array)

        for i, sample in enumerate(self.samples):
            sample['structural'] = normalized[i].astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]

        return {
            'sequence': torch.tensor(sample['sequence'], dtype=torch.long),
            'structural': torch.tensor(sample['structural'], dtype=torch.float),
            'length': torch.tensor(sample['length'], dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }


def collate_hybrid(batch):
    """Collate function for hybrid model"""
    max_len = max(item['length'].item() for item in batch)
    max_len = min(max_len, 4096)  # Cap at max length

    sequences = []
    for item in batch:
        seq = item['sequence'][:max_len]
        if len(seq) < max_len:
            seq = F.pad(seq, (0, max_len - len(seq)))
        sequences.append(seq)

    return {
        'sequence': torch.stack(sequences),
        'structural': torch.stack([item['structural'] for item in batch]),
        'length': torch.stack([item['length'].clamp(max=max_len) for item in batch]),
        'label': torch.stack([item['label'] for item in batch])
    }


# ============================================================================
# TRAINING
# ============================================================================

class HybridTrainer:
    """
    Trainer for Hybrid CNN-LSTM model
    """

    def __init__(self, model: nn.Module, device: str = 'cuda',
                 learning_rate: float = 1e-3, weight_decay: float = 1e-4):
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )

        self.criterion = nn.CrossEntropyLoss()

        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'val_auc': [], 'val_f1': []
        }

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            sequence = batch['sequence'].to(self.device)
            structural = batch['structural'].to(self.device)
            lengths = batch['length'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(sequence, structural, lengths)
            loss = self.criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * sequence.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += sequence.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []

        for batch in loader:
            sequence = batch['sequence'].to(self.device)
            structural = batch['structural'].to(self.device)
            lengths = batch['length'].to(self.device)
            labels = batch['label'].to(self.device)

            logits = self.model(sequence, structural, lengths)
            loss = self.criterion(logits, labels)

            total_loss += loss.item() * sequence.size(0)

            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        accuracy = (all_preds == all_labels).mean()
        avg_loss = total_loss / len(all_labels)

        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.5

        f1 = f1_score(all_labels, all_preds, average='weighted')

        return avg_loss, accuracy, auc, f1, all_preds, all_labels

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int = 100, patience: int = 15):
        """Full training loop"""

        best_val_auc = 0
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc, val_auc, val_f1, _, _ = self.evaluate(val_loader)

            # Scheduler step
            self.scheduler.step(val_auc)

            # Log
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['val_auc'].append(val_auc)
            self.history['val_f1'].append(val_f1)

            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"  Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}")

            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self.history

    def get_attention_analysis(self, loader: DataLoader) -> List[Dict]:
        """Get attention weights for analysis"""
        self.model.eval()
        results = []

        with torch.no_grad():
            for batch in loader:
                sequence = batch['sequence'].to(self.device)
                structural = batch['structural'].to(self.device)
                lengths = batch['length'].to(self.device)
                labels = batch['label'].to(self.device)

                logits, token_scores, attn_weights = self.model(
                    sequence, structural, lengths, return_attention=True
                )

                for i in range(sequence.size(0)):
                    results.append({
                        'label': labels[i].item(),
                        'prediction': logits[i].argmax().item(),
                        'confidence': F.softmax(logits[i], dim=0).max().item(),
                        'token_scores': token_scores[i].cpu().numpy(),
                        'attention_weights': attn_weights[i].cpu().numpy()
                    })

        return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_attention_heatmap(attention_weights: np.ndarray, tokens: List[str],
                           title: str = "Attention Weights", save_path: Optional[str] = None):
    """Plot attention heatmap"""
    plt.figure(figsize=(12, 4))

    # Limit to first 100 tokens for visualization
    n = min(100, len(attention_weights))
    weights = attention_weights[:n]
    labels = tokens[:n] if tokens else [str(i) for i in range(n)]

    plt.bar(range(n), weights, color='steelblue', alpha=0.7)
    plt.xlabel('Token Position')
    plt.ylabel('Attention Weight')
    plt.title(title)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_curves(history: Dict, save_path: Optional[str] = None):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()

    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy Curves')
    axes[0, 1].legend()

    # AUC
    axes[1, 0].plot(history['val_auc'], label='Validation AUC', color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC-ROC')
    axes[1, 0].set_title('Validation AUC')
    axes[1, 0].legend()

    # F1
    axes[1, 1].plot(history['val_f1'], label='Validation F1', color='orange')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].set_title('Validation F1 Score')
    axes[1, 1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution"""

    config = {
        'data_dir': r'D:\IIT K RTL\Dataset_Hardware_Trojan\FINAL_DATA\FINAL_DATA\datasets',
        'batch_size': 16,
        'vocab_size': 5000,
        'embed_dim': 128,
        'hidden_dim': 256,
        'num_lstm_layers': 2,
        'num_attention_heads': 8,
        'max_seq_len': 2048,
        'dropout': 0.3,
        'learning_rate': 1e-3,
        'epochs': 100,
        'patience': 15,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("=" * 70)
    print("Hardware Trojan Detection using Hybrid CNN-LSTM with Attention")
    print("=" * 70)
    print(f"\nDevice: {config['device']}")

    # Feature extractor
    feature_extractor = VerilogFeatureExtractor(
        vocab_size=config['vocab_size'],
        max_seq_len=config['max_seq_len']
    )

    # Load dataset
    print("\nLoading dataset...")
    dataset = HybridTrojanDataset(
        config['data_dir'],
        feature_extractor,
        config['max_seq_len']
    )

    # Split
    indices = list(range(len(dataset)))
    labels = dataset.labels

    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=42
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.15,
        stratify=[labels[i] for i in train_idx],
        random_state=42
    )

    # Create subsets
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, collate_fn=collate_hybrid
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        collate_fn=collate_hybrid
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'],
        collate_fn=collate_hybrid
    )

    print(f"\nDataset splits:")
    print(f"  Training: {len(train_idx)}")
    print(f"  Validation: {len(val_idx)}")
    print(f"  Test: {len(test_idx)}")

    # Create model
    model = HybridCNNLSTMTrojanDetector(
        vocab_size=len(feature_extractor.token_to_id),
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_structural_features=len(feature_extractor.structural_features),
        num_lstm_layers=config['num_lstm_layers'],
        num_attention_heads=config['num_attention_heads'],
        num_classes=2,
        dropout=config['dropout']
    )

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    trainer = HybridTrainer(
        model,
        device=config['device'],
        learning_rate=config['learning_rate']
    )

    print("\nStarting training...")
    history = trainer.fit(
        train_loader, val_loader,
        epochs=config['epochs'],
        patience=config['patience']
    )

    # Final evaluation
    print("\n" + "=" * 70)
    print("Final Evaluation on Test Set")
    print("=" * 70)

    test_loss, test_acc, test_auc, test_f1, preds, labels_arr = trainer.evaluate(test_loader)

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUC-ROC: {test_auc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(labels_arr, preds, target_names=['Clean', 'Trojaned']))

    # Confusion matrix
    cm = confusion_matrix(labels_arr, preds)
    print("\nConfusion Matrix:")
    print(cm)

    # Save model
    save_path = os.path.join(os.path.dirname(config['data_dir']), 'hybrid_cnn_lstm_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': feature_extractor.token_to_id,
        'scaler_mean': dataset.scaler.mean_,
        'scaler_scale': dataset.scaler.scale_,
        'config': config,
        'history': history
    }, save_path)
    print(f"\nModel saved to: {save_path}")

    # Plot training curves
    plot_training_curves(history)

    return model, trainer, history


if __name__ == "__main__":
    model, trainer, history = main()
