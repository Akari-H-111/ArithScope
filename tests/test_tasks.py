"""Task generators and their Bayes oracles versus brute-force recomputation."""
import unittest
from collections import Counter
from importlib.util import find_spec

HAS_TORCH = find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from arithscope.tasks import AssociativeRecall, PeriodicDenoising, TaggedDenoising


def brute_vote(symbols, groups, truth, vocab, observed=None):
    """Expected accuracy of a causal plurality vote within groups, ties split uniformly.

    With no votes at all, every symbol ties, so the expected accuracy is ``1 / vocab``.
    """
    out = []
    for t in range(len(symbols)):
        counts = Counter(symbols[j] for j in range(t + 1)
                         if groups[j] == groups[t] and (observed is None or observed[j]))
        if not counts:
            out.append(1.0 / vocab)
            continue
        best = max(counts.values())
        winners = [s for s, c in counts.items() if c == best]
        out.append(1.0 / len(winners) if truth[t] in winners else 0.0)
    return out


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TaskOracleTests(unittest.TestCase):
    def test_periodic_oracle_matches_brute_force(self):
        task = PeriodicDenoising(periods=(5, 7), vocab_size=6, noise=0.6)
        batch = task.sample(6, 40, torch.Generator().manual_seed(1))
        for b in range(6):
            p = int(batch.group[b])
            expected = brute_vote(batch.tokens[b].tolist(), [t % p for t in range(40)], batch.targets[b].tolist(), 6)
            self.assertTrue(torch.allclose(batch.oracle[b], torch.tensor(expected)), b)
            self.assertTrue(all(batch.targets[b, t] == batch.targets[b, t % p] for t in range(40)))

    def test_alias_vote_oracle_matches_brute_force(self):
        task = PeriodicDenoising(periods=(24,), vocab_size=5, noise=0.5, query_rate=0.3)
        batch = task.sample(4, 120, torch.Generator().manual_seed(2))
        alias = task.vote_oracle(batch, 12)
        for b in range(4):
            observed = batch.observed[b].tolist()
            expected = brute_vote(batch.tokens[b].tolist(), [t % 12 for t in range(120)], batch.targets[b].tolist(),
                                  5, observed)
            self.assertTrue(torch.allclose(alias[b], torch.tensor(expected)), b)

    def test_query_positions_carry_no_evidence(self):
        task = PeriodicDenoising(periods=(24,), query_rate=0.25)
        batch = task.sample(8, 256, torch.Generator().manual_seed(3))
        self.assertTrue(torch.equal(batch.loss_mask, ~batch.observed))
        self.assertTrue(bool((batch.tokens[batch.loss_mask] == task.vocab_size).all()))
        self.assertEqual(task.model_vocab, task.vocab_size + 1)
        # The full-period vote equals the stored Bayes oracle.
        self.assertTrue(torch.allclose(task.vote_oracle(batch, 24), batch.oracle))

    def test_tagged_oracle_matches_brute_force(self):
        task = TaggedDenoising(num_tags=4, vocab_size=5, noise=0.5)
        batch = task.sample(5, 30, torch.Generator().manual_seed(4))
        for b in range(5):
            tags = (batch.tokens[b] // 5).tolist()
            symbols = (batch.tokens[b] % 5).tolist()
            expected = brute_vote(symbols, tags, batch.targets[b].tolist(), 5)
            self.assertTrue(torch.allclose(batch.oracle[b], torch.tensor(expected)), b)

    def test_mqar_layout(self):
        task = AssociativeRecall(num_keys=40, num_values=8)
        batch = task.sample(3, 32, torch.Generator().manual_seed(5))
        for b in range(3):
            toks = batch.tokens[b].tolist()
            table = {toks[i]: toks[i + 1] for i in range(0, 16, 2)}
            self.assertEqual(len(table), 8)  # keys are distinct
            for t in range(16, 32, 2):
                self.assertTrue(batch.loss_mask[b, t])
                self.assertEqual(batch.targets[b, t].item(), table[toks[t]])
                self.assertEqual(toks[t + 1], table[toks[t]])
            self.assertEqual(int(batch.loss_mask[b].sum()), 8)

    def test_samplers_are_deterministic(self):
        for task in (PeriodicDenoising(periods=(7, 11, 13)), TaggedDenoising(), AssociativeRecall()):
            a = task.sample(3, 64, torch.Generator().manual_seed(9))
            b = task.sample(3, 64, torch.Generator().manual_seed(9))
            self.assertEqual(a.digest(), b.digest())


if __name__ == "__main__":
    unittest.main()
