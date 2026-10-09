# Loss Recovered (Methods → Loss Recovered; Results → Loss Recovered)

> **Status: code not yet imported.** The paper's Loss Recovered numbers and the
> SAE baseline were produced on a different cluster. This directory is a
> placeholder that pins the protocol and lists what must be copied in.

## Protocol (from the paper)

- Target: Gemma-3-12B-IT (`soar_nla.hub.GEMMA3_12B`), residual stream at layer 32.
- NLA: official release, `kitft/nla-gemma3-12b-L32-{av,ar}` (`soar_nla.hub.KITFT_L32_AV/AR`).
  Each AR reconstruction is rescaled to the norm of the original activation.
- Data: 1,543 Alpaca generations with at least 100 tokens. These are the
  shared Alpaca rollouts from `soar_nla.alpaca` (2,000 greedy Gemma-3-12B
  generations); exactly 1,543 of them have `assistant_token_count >= 100`.
  Confirm this is the same set before relying on it.
- Metric (SAEBench convention): `LR = (H_NLA − H_0) / (H_orig − H_0)`, with
  `H_0` the cross-entropy loss after zero-ablating the activations.
- Three settings: reconstruct tokens 1–100; reconstruct tokens 1–50;
  reconstruct tokens 51–100 while keeping tokens 1–50 clean.
- Baseline: a 16k-width SAE on layer 32 with mean L0 = 20.82, evaluated the
  same way with norm rescaling.

| Paper claim | Value |
|---|---|
| NLA, tokens 51–100 (1–50 clean) | 90.88% |
| NLA, tokens 1–50 | 64.37% |
| NLA, tokens 1–100 | 69.95% |
| SAE (16k, L0 = 20.82), tokens 51–100, norm-rescaled | 97.58% |

## Needed from the other cluster

1. The loss-recovered script(s): window definitions, whether "tokens 1–100"
   indexes assistant-response tokens or the full chat-rendered sequence,
   padding/attention-mask handling, the patching hook and CE scoring.
2. The rollouts file used (or confirmation it equals `soar_nla.alpaca` output).
3. The SAE identity (release, `sae_id`, which L0 variant gives mean L0 = 20.82),
   the SAE patching code with and without norm rescaling, and its environment
   (e.g. `sae_lens` version).
4. Summary outputs (per-window clean / NLA / zero-ablation CE) and any figure
   or table script.

Put the code in this subpackage following the repository conventions
(argparse CLI, pinned checkpoints via `soar_nla.hub`, outputs under
`soar_nla.paths.results_dir("loss_recovered")`).
