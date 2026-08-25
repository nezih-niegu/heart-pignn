import torch

from heart_pignn.attention import BatchPrototypeMemory, CardiacAttentionBridge
from heart_pignn.graph import build_heart_graph
from heart_pignn.heuristics import N_REGIMES, N_RULE_FEATURES
from heart_pignn.losses import BeatLoss
from heart_pignn.model import ModelConfig, PIGNNBeatClassifier

SMALL = ModelConfig(hidden_dim=16, msg_dim=16, node_emb_dim=8, n_layers=1, graph_steps=4)


def make_batch(b=6, t=360):
    return (
        torch.randn(b, 1, t),
        torch.rand(b, N_RULE_FEATURES),
        torch.randint(0, N_REGIMES, (b,)),
    )


def test_graph_is_well_formed():
    g = build_heart_graph()
    assert g.n_nodes == 24
    assert g.edge_index.shape[0] == 2
    assert int(g.edge_index.max()) < g.n_nodes
    assert g.edge_attr.shape == (g.n_edges, 3)


def test_forward_shapes():
    model = PIGNNBeatClassifier(SMALL)
    x, rule, regime = make_batch()
    out = model(x, rule, regime, return_attention=True)
    assert out["logits"].shape == (6, 5)
    assert out["rule_logits"].shape == (6, 3)
    assert out["latent"].shape == (6, SMALL.hidden_dim)
    assert out["node_importance"].shape == (6, 24)
    assert out["vm"].shape == (6, SMALL.graph_steps, 24)


def test_attention_weights_sum_to_one():
    bridge = CardiacAttentionBridge(hidden_dim=16)
    states = torch.randn(4, 5, 24, 16)
    out = bridge(states)
    assert torch.allclose(out["node_attention"].sum(-1), torch.ones(4, 5), atol=1e-5)
    assert torch.allclose(out["time_attention"].sum(-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(out["node_importance"].sum(-1), torch.ones(4), atol=1e-5)


def test_prototypes_update_only_for_regimes_present():
    mem = BatchPrototypeMemory(hidden_dim=8, n_regimes=4)
    z = torch.randn(10, 8)
    regimes = torch.zeros(10, dtype=torch.long)
    mem.update(z, regimes)
    assert bool(mem.initialized[0])
    assert not bool(mem.initialized[1:].any())
    sim = mem.similarity(z)
    assert sim.shape == (10, 4)
    assert torch.allclose(sim[:, 1:], torch.zeros(10, 3))


def test_prototypes_frozen_in_eval():
    model = PIGNNBeatClassifier(SMALL)
    x, rule, regime = make_batch()
    model.eval()
    before = model.prototypes.prototypes.clone()
    model(x, rule, regime)
    assert torch.equal(before, model.prototypes.prototypes)


def test_backward_reaches_graph_and_rule_paths():
    model = PIGNNBeatClassifier(SMALL)
    x, rule, regime = make_batch()
    out = model(x, rule, regime)
    losses = BeatLoss(rule_w=0.3, phys_w=0.05)(
        out, torch.randint(0, 5, (6,)), torch.rand(6, 3).round()
    )
    losses["loss"].backward()

    msg_grad = model.encoder.cells[0].msg_net[0].weight.grad
    film_grad = model.film.encoder[0].weight.grad
    assert msg_grad is not None and msg_grad.abs().sum() > 0
    assert film_grad is not None and film_grad.abs().sum() > 0


def test_rule_head_reads_pre_film_latent():
    """The rule head must not be able to copy its own input via FiLM."""
    model = PIGNNBeatClassifier(SMALL).eval()
    x, _, regime = make_batch()
    a = model(x, torch.zeros(6, N_RULE_FEATURES), regime)["rule_logits"]
    b = model(x, torch.ones(6, N_RULE_FEATURES), regime)["rule_logits"]
    assert torch.allclose(a, b, atol=1e-6)
