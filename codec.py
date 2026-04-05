"""
ArithmeticCodec / Dense Bit Packer

Due to the autoregressive nature of the FSQEntropyModel (p(z_i | z_{<i})),
standard arithmetic coding libraries like `torchac` cannot be efficiently used 
in Python during decode (since they expect the full CDFs tensor upfront, which
cannot be computed without the previously decoded symbols). 

To avoid the massive overhead of encoding each symbol as a separate bitstream 
(which costs ~4 bytes per slot!), this module implements a dense bit-packer.
It losslessly packs the active 12-bit FSQ indices into a continuous byte array.
It relies purely on the gate network's sparsity to save bits (uniform coding).

The theoretical arithmetic-coded bits are still estimated by the entropy model
and logged as `entropy_bits`, but the actual `payload` produced here is a 
strictly fair, fully decodable uniform bitstream that you can write to disk.

Payload wire format (all big-endian):
    [8 bytes]   gate_mask  — packed uint64, bit i = slot i active
    [X bytes]   dense_bits — the 12-bit indices of ONLY the active slots, 
                             packed continuously without padding.
"""

import struct
import math
import torch

# ──────────────────────────────────────────────────────────────────────────────
# Gate mask helpers
# ──────────────────────────────────────────────────────────────────────────────

def pack_gate_mask(gate_mask: torch.Tensor) -> bytes:
    """
    Pack a 1-D bool tensor of length Q (≤64) into 8 bytes.
    Bit i of the uint64 = gate_mask[i].
    """
    val = 0
    for i, g in enumerate(gate_mask.tolist()):
        if g:
            val |= (1 << i)
    return struct.pack(">Q", val)


def unpack_gate_mask(data: bytes, Q: int) -> torch.Tensor:
    """Inverse of pack_gate_mask. Returns bool tensor [Q]."""
    val = struct.unpack(">Q", data[:8])[0]
    return torch.tensor([(val >> i) & 1 for i in range(Q)], dtype=torch.bool)


# ──────────────────────────────────────────────────────────────────────────────
# Bit packing helpers (Dense 12-bit packing)
# ──────────────────────────────────────────────────────────────────────────────

def _bits_for_uniform(codebook_size: int) -> int:
    return math.ceil(math.log2(max(codebook_size, 2)))

def _pack_bits(symbols: list[int], bits_per_sym: int) -> bytes:
    """Pack a list of integers (each taking `bits_per_sym` bits) into bytes."""
    bit_string = ""
    for sym in symbols:
        bit_string += format(sym, f"0{bits_per_sym}b")
    
    # Pad to full bytes
    padding = (8 - len(bit_string) % 8) % 8
    bit_string += "0" * padding
    
    # Convert to bytes
    byte_array = bytearray()
    for i in range(0, len(bit_string), 8):
        byte_array.append(int(bit_string[i:i+8], 2))
        
    return bytes(byte_array)

def _unpack_bits(data: bytes, num_symbols: int, bits_per_sym: int) -> list[int]:
    """Unpack `num_symbols` integers (each `bits_per_sym` bits) from bytes."""
    bit_string = "".join(format(b, "08b") for b in data)
    
    symbols = []
    for i in range(num_symbols):
        start = i * bits_per_sym
        end = start + bits_per_sym
        symbols.append(int(bit_string[start:end], 2))
        
    return symbols


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def compress(
    flat_indices: torch.Tensor,
    gate_mask: torch.Tensor,
    entropy_model,
) -> bytes:
    """
    Encode a single document's active FSQ indices to a dense byte string.
    
    Args:
        flat_indices:  [1, Q] long — flat (mixed-radix) FSQ index per slot
        gate_mask:     [1, Q] bool
        entropy_model: Used to find the codebook size.

    Returns:
        payload: bytes
    """
    device = flat_indices.device
    mask = gate_mask[0]
    
    # 1. Pack the gate mask (8 bytes)
    payload = pack_gate_mask(mask)
    
    # 2. Extract only the active symbols
    active_symbols = flat_indices[0][mask].tolist()
    
    # 3. Dense pack the active symbols
    if active_symbols:
        C = entropy_model.codebook_size if entropy_model is not None else 4096
        bits_per_sym = _bits_for_uniform(C)
        payload += _pack_bits(active_symbols, bits_per_sym)
        
    return payload


def decompress(
    payload: bytes,
    Q: int,
    entropy_model,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decode a dense byte string back to FSQ indices + gate mask.

    Args:
        payload:       bytes produced by compress()
        Q:             number of FSQ slots
        entropy_model: Used to find the codebook size.
        device:        torch device

    Returns:
        flat_indices: [1, Q] long
        gate_mask:    [1, Q] bool
    """
    # 1. Unpack the gate mask
    mask = unpack_gate_mask(payload[:8], Q)
    num_active = mask.sum().item()
    
    # 2. Unpack the active symbols
    flat_indices = torch.zeros(1, Q, dtype=torch.long)
    if num_active > 0:
        C = entropy_model.codebook_size if entropy_model is not None else 4096
        bits_per_sym = _bits_for_uniform(C)
        active_symbols = _unpack_bits(payload[8:], num_active, bits_per_sym)
        
        # 3. Scatter active symbols back into their slots
        # (Inactive slots remain 0. They will be ignored by the decoder anyway
        # because the gate_mask zeros them out in the attention mask).
        flat_indices[0][mask] = torch.tensor(active_symbols, dtype=torch.long)
        
    return flat_indices.to(device), mask.unsqueeze(0).to(device)


def payload_bits(payload: bytes) -> float:
    """Return bit count of a payload produced by compress()."""
    return len(payload) * 8


def is_entropy_coded() -> bool:
    """Always False now. Using dense uniform bit packing."""
    return False
