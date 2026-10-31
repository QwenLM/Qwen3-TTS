"""CPU regressions: python -m unittest discover -s tests -v.

No pretrained weights, audio, accelerator or network access is required.
The SFT tensor block is executed from its AST to avoid importing the CLI's
dataset/Accelerate stack or starting its training loop. The code predictor
test uses the actual model class with a tiny randomly initialized decoder.
"""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from transformers.loss.loss_utils import ForCausalLMLoss
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerCodePredictorConfig
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerCodePredictorModelForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]


def sft_tensor_block():
    tree = ast.parse((ROOT / 'finetuning/sft_12hz.py').read_text(encoding='utf-8'))
    train = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
    block = next(n for n in ast.walk(train) if isinstance(n, ast.With)
                 and any(isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name)
                         and s.targets[0].id == 'input_text_ids' for s in n.body))
    statements = block.body
    start = next(i for i, s in enumerate(statements) if isinstance(s, ast.Assign)
                 and isinstance(s.targets[0], ast.Name) and s.targets[0].id == 'input_text_ids')
    end = next(i for i, s in enumerate(statements) if isinstance(s, ast.Assign)
               and isinstance(s.targets[0], ast.Name) and s.targets[0].id == 'loss')
    return compile(ast.Module(body=statements[start:end + 1], type_ignores=[]), 'sft_12hz.py', 'exec')


class TinyTalker(nn.Module):
    def __init__(self, text_width):
        super().__init__()
        self.model = SimpleNamespace(text_embedding=nn.Embedding(32, text_width), codec_embedding=nn.Embedding(32, 8))
        self.text_projection = nn.Linear(text_width, 8)
        embeddings = nn.ModuleList([nn.Embedding(32, 8) for _ in range(15)])
        self.code_predictor = SimpleNamespace(get_input_embeddings=lambda: embeddings)
        self.head = nn.Linear(8, 32)

    def forward(self, inputs_embeds, attention_mask, labels, output_hidden_states):
        self.received_labels = labels
        self.received_mask = attention_mask
        self.hidden = inputs_embeds.cumsum(1)  # inexpensive causal decoder
        self.logits = self.head(self.hidden)
        return SimpleNamespace(loss=ForCausalLMLoss(self.logits, labels, 32),
                               hidden_states=((self.hidden,),))

    def forward_sub_talker_finetune(self, codes, hidden):
        self.residual_codes, self.residual_hidden = codes, hidden
        return None, hidden.sum() * 0


class FinetuningAlignment(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def batch(self, text_width=8):
        talker = TinyTalker(text_width)
        codec_mask = torch.zeros(2, 12, dtype=torch.bool)
        codec_mask[0, 7:11] = True
        codec_mask[1, 7:9] = True  # unequal lengths and padding
        labels = torch.full((2, 12), -100, dtype=torch.long)
        labels[codec_mask] = torch.arange(6) + 1
        return dict(model=SimpleNamespace(talker=talker), input_ids=torch.randint(0, 32, (2, 12, 2)),
                    codec_ids=torch.randint(0, 32, (2, 12, 16)), codec_mask=codec_mask,
                    codec_0_labels=labels, text_embedding_mask=torch.ones(2, 12, 1),
                    codec_embedding_mask=torch.ones(2, 12, 1), attention_mask=torch.ones(2, 12),
                    speaker_embedding=torch.zeros(2, 8))

    def execute(self, batch):
        scope = dict(batch)
        exec(sft_tensor_block(), scope)
        return scope

    def test_talker_uses_exactly_one_temporal_shift(self):
        batch = self.batch()
        self.execute(batch)
        talker = batch['model'].talker
        torch.testing.assert_close(talker.received_labels, batch['codec_0_labels'])
        torch.testing.assert_close(talker.received_mask, batch['attention_mask'])
        expected = F.cross_entropy(talker.logits[:, :-1].reshape(-1, 32), batch['codec_0_labels'][:, 1:].reshape(-1))
        actual = ForCausalLMLoss(talker.logits, talker.received_labels, 32)
        torch.testing.assert_close(actual, expected)

    def test_projection_supports_unequal_text_and_talker_width(self):
        batch = self.batch(text_width=6)
        result = self.execute(batch)  # also catches undefined self.model in the CLI
        self.assertTrue(torch.isfinite(result['loss']))
        result['loss'].backward()
        self.assertGreater(batch['model'].talker.text_projection.weight.grad.abs().sum().item(), 0)

    def test_residual_condition_precedes_target_frame(self):
        batch = self.batch()
        self.execute(batch)
        talker = batch['model'].talker
        expected = torch.stack([talker.hidden[b, t - 1] for b, t in batch['codec_mask'].nonzero().tolist()])
        torch.testing.assert_close(talker.residual_hidden, expected)
        torch.testing.assert_close(talker.residual_codes, batch['codec_ids'][batch['codec_mask']])

    def test_current_residual_codes_cannot_change_current_condition(self):
        batch = self.batch()
        self.execute(batch)
        before = batch['model'].talker.residual_hidden.detach().clone()
        batch['codec_ids'][0, 7, 1:] = (batch['codec_ids'][0, 7, 1:] + 1) % 32
        self.execute(batch)
        after = batch['model'].talker.residual_hidden
        torch.testing.assert_close(before[0], after[0])
        self.assertFalse(torch.allclose(before[1], after[1]))

    def test_residual_loss_supervises_each_codebook_without_shift(self):
        config = Qwen3TTSTalkerCodePredictorConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=8,
            num_code_groups=16, use_cache=False)
        config._attn_implementation = 'eager'
        model = Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(
            config, talker_config=SimpleNamespace(hidden_size=16)).eval()
        # Exercise noncontiguous targets and ignore_index as well as all 15 heads.
        labels = torch.arange(30).reshape(15, 2).t()
        labels[1, 3] = -100
        self.assertFalse(labels.is_contiguous())
        inputs = torch.randn(2, 16, 16)
        output = model.forward_finetune(inputs_embeds=inputs, labels=labels, use_cache=False)
        expected = F.cross_entropy(output.logits.reshape(-1, 32), labels.reshape(-1))
        torch.testing.assert_close(output.loss, expected)
        output.logits.retain_grad()
        output.loss.backward()
        independent = output.logits.detach().clone().requires_grad_()
        F.cross_entropy(independent.reshape(-1, 32), labels.reshape(-1)).backward()
        torch.testing.assert_close(output.logits.grad, independent.grad)
        for head in model.lm_head:
            self.assertGreater(head.weight.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
