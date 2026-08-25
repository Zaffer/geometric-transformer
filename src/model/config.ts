// Model configuration for the pure GPT-2 style transformer.
// "Pure" means: token embedding + learned positional embedding,
// N x [Norm, causal multi-head attention, residual, Norm, MLP, residual],
// final Norm, unembedding, softmax. The toggles below let you remove or
// swap parts to reach the attention-only toy models or the modern variants.

export type Activation = 'gelu' | 'relu';
export type NormKind = 'layernorm' | 'rmsnorm' | 'none';

export interface ModelConfig {
  nLayer: number;
  nHead: number;
  dModel: number;
  vocab: number;
  seqLen: number; // number of tokens to sort
  nCtx: number; // context length = 2 * seqLen - 1
  mlpRatio: number;
  tieWeights: boolean; // GPT-2 ties wte and the unembedding
  activation: Activation; // GPT-2: gelu (tanh approximation)
  norm: NormKind; // GPT-2: layernorm
  mlp: boolean; // false = attention-only blocks
}

export interface ConfigInput {
  nLayer: number;
  nHead: number;
  dModel: number;
  seqLen: number;
  tieWeights: boolean;
  activation?: Activation;
  norm?: NormKind;
  mlp?: boolean;
}

export const VOCAB = 3; // tokens A, B, C for the sort task
export const TOKEN_NAMES = ['A', 'B', 'C'];

export function makeConfig(p: ConfigInput): ModelConfig {
  if (p.dModel % p.nHead !== 0) {
    throw new Error(`dModel (${p.dModel}) must be divisible by nHead (${p.nHead})`);
  }
  return {
    nLayer: p.nLayer,
    nHead: p.nHead,
    dModel: p.dModel,
    vocab: VOCAB,
    seqLen: p.seqLen,
    nCtx: 2 * p.seqLen - 1,
    mlpRatio: 4,
    tieWeights: p.tieWeights,
    activation: p.activation ?? 'gelu',
    norm: p.norm ?? 'layernorm',
    mlp: p.mlp ?? true,
  };
}

// The "micro" default the user chose: 2 layers, 2 heads, width 16, pure GPT-2.
export const MICRO: Required<ConfigInput> = {
  nLayer: 2,
  nHead: 2,
  dModel: 16,
  seqLen: 6,
  tieWeights: true,
  activation: 'gelu',
  norm: 'layernorm',
  mlp: true,
};
