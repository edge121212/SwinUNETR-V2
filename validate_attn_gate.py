"""CPU-only validation for the attention-gate SwinUNETR vendored copy.

Checks (no GPU, does not touch the running baseline):
  1. attn_gate_levels=() -> state_dict keys identical to stock MONAI SwinUNETR (use_v2=True).
  2. Parameter counts: off == stock (72,763,845); gated adds a few small modules.
  3. Forward + backward smoke test on a tiny 32^3 input for off / full / high-res-only configs.
"""

import torch
from monai.networks.nets import SwinUNETR as StockSwinUNETR

from models.swin_unetr import SwinUNETR as LocalSwinUNETR

COMMON = dict(in_channels=2, out_channels=3, feature_size=48, use_v2=True)


def build_local(levels):
    return LocalSwinUNETR(**COMMON, attn_gate_levels=levels)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def main():
    torch.manual_seed(0)

    stock = StockSwinUNETR(**COMMON)
    off = build_local(())
    full = build_local((1, 2, 3, 4, 5))
    hires = build_local((1, 2, 3))

    # 1. state_dict key equivalence (off vs stock)
    stock_keys = set(stock.state_dict().keys())
    off_keys = set(off.state_dict().keys())
    assert stock_keys == off_keys, (
        f"OFF keys differ from stock! "
        f"only_stock={list(stock_keys - off_keys)[:5]} only_off={list(off_keys - stock_keys)[:5]}"
    )
    print("[1] state_dict keys: OFF == stock  OK  (%d keys)" % len(off_keys))

    # off should also load a stock state_dict cleanly (baseline checkpoint compat)
    off.load_state_dict(stock.state_dict())
    print("    OFF.load_state_dict(stock)  OK  (baseline checkpoint compatible)")

    # 2. param counts
    p_stock, p_off, p_full, p_hires = map(count_params, (stock, off, full, hires))
    assert p_off == p_stock, f"OFF param count {p_off} != stock {p_stock}"
    gate_keys_full = [k for k in full.state_dict() if "attn_gate" in k]
    gate_keys_hires = [k for k in hires.state_dict() if "attn_gate" in k]
    print("[2] params  stock=%d  off=%d  full=%d (+%d)  hires=%d (+%d)"
          % (p_stock, p_off, p_full, p_full - p_stock, p_hires, p_hires - p_stock))
    print("    attn_gate tensors:  full=%d  hires=%d" % (len(gate_keys_full), len(gate_keys_hires)))
    assert len(gate_keys_full) == 5 * 6, "expected 3 convs x (w+b) x 5 levels"   # w_x,w_g,psi each have weight+bias
    assert len(gate_keys_hires) == 3 * 6, "expected 3 convs x (w+b) x 3 levels"

    # 3. forward + backward smoke (64^3, CPU; 32^3 is too small -> 1^3 bottleneck breaks InstanceNorm)
    x = torch.randn(1, 2, 64, 64, 64)
    for name, model in [("off", off), ("full", full), ("hires", hires)]:
        model.train()
        y = model(x)
        assert y.shape == (1, 3, 64, 64, 64), f"{name} bad output shape {y.shape}"
        loss = y.float().pow(2).mean()
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        gate_grad = None
        if name != "off":
            gate_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for n, p in model.named_parameters() if "attn_gate" in n
            )
        print("[3] %-5s forward %s  backward grads=%s  gate_grads=%s"
              % (name, tuple(y.shape), has_grad, gate_grad))
        assert has_grad
        if name != "off":
            assert gate_grad, f"{name}: attention gate received no gradient"

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
