# Qwen3 Inference Conditions For Phase 1.5 Rerun

Date: 2026-05-14 UTC

## Sources Checked

- Qwen/Qwen3-8B Hugging Face model card: `https://huggingface.co/Qwen/Qwen3-8B`
- Qwen quickstart docs: `https://qwen.readthedocs.io/en/stable/getting_started/quickstart.html`
- Qwen function-calling docs: `https://qwen.readthedocs.io/en/stable/framework/function_call.html`
- Existing successful prime-rl base rollouts: `runs/rollouts/prime_swe_grep_Qwen_Qwen3_8B/summary.json`

## Relevant Guidance

Qwen3 supports thinking and non-thinking modes. Qwen's docs state that thinking mode
is the default, but non-thinking mode is available through `enable_thinking=False`.
For function calling, Qwen's docs recommend Hermes-style tool use and show a
Qwen3-8B no-thinking function-calling example with:

- `temperature=0.7`
- `top_p=0.8`
- `max_tokens=512`
- `repetition_penalty=1.05`
- `chat_template_kwargs={"enable_thinking": False}`

Qwen's quickstart/best-practices page also warns against greedy decoding in thinking
mode, and recommends non-thinking mode sampling with:

- `temperature=0.7`
- `top_p=0.8`
- `top_k=20`
- `min_p=0`

Qwen3 supports normal short contexts up to 32,768 tokens without YaRN. The successful
prime-rl swe-grep base rollout set used much longer prompt contexts than our failed
HF smoke: mean final input tokens were about `1292.79`, max final input tokens were
`2212`, and max total input tokens across the tool loop were `6404`.

## Previous Mismatches

The failed HF Phase 1.5 smoke used:

- `temperature=0.0`, i.e. greedy decoding
- `max_prefix_tokens=512`, truncating the tool-use system prompt and multi-turn
  context
- float32 steering after the precision fix, which made linearization valid but did
  not fix the behavioral degeneration

The prompt truncation is likely load-bearing: the first turn in successful prime-rl
base swe-grep rollouts used about `1044` prompt tokens, while the failed HF path
forced every turn down to `512`.

## Rerun Settings

The Phase 1.5 rerun uses:

- `dtype=float32`
- `enable_thinking=false`
- `temperature=0.7`
- `top_p=0.8`
- `top_k=20`
- `min_p=0.0`
- `repetition_penalty=1.05`
- `max_completion_tokens=512`
- `max_prefix_tokens=8192`

This combines Qwen's no-thinking function-calling recommendations with enough
context budget to preserve the full swe-grep tool-use prompt and tool-result history.
