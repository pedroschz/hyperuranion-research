1. model.py: Keep the forced-open gates during stage 1, but make it continuous:
   `decoder_gates = torch.ones_like(gates) if rate_scale == 0.0 else gates`
2. loss.py: Add `gate_warmup_loss` so the gate_net outputs 1.0 during stage 1.
3. config.py: Increase LEARNING_RATE from 2e-5 to 2e-4. (This is the main reason distortion is stuck at 5.0).
