import unittest

import torch
import torch.nn.functional as F

from rul.models import tfg_mamba as M


def _scan_inputs(b: int = 3, length: int = 13, d: int = 8, n: int = 4, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(b, length, d, generator=g)
    delta = F.softplus(torch.randn(b, length, d, generator=g))
    A = -torch.exp(torch.randn(d, n, generator=g))
    B = torch.randn(b, length, n, generator=g)
    C = torch.randn(b, length, n, generator=g)
    D = torch.randn(d, generator=g)
    return u, delta, A, B, C, D


class ScanTest(unittest.TestCase):
    def test_logdepth_equals_sequential(self) -> None:
        for length in (1, 2, 5, 13, 32, 33):
            args = _scan_inputs(length=length)
            ref = M.selective_scan_sequential(*args)
            fast = M.selective_scan_logdepth(*args)
            torch.testing.assert_close(fast, ref, rtol=1e-4, atol=1e-5)

    def test_zoh_matches_matrix_formula(self) -> None:
        # Eq. (19): B_bar = (dA)^-1 (exp(dA) - I) dB with diagonal A.
        delta = torch.tensor([[[0.3]]], dtype=torch.float64)
        A = torch.tensor([[-0.5, -2.0, -7.0]], dtype=torch.float64)
        B = torch.tensor([[[1.0, -2.0, 0.5]]], dtype=torch.float64)
        A_bar, B_bar = M.zoh_discretize(delta, A, B)
        dA = torch.diag(0.3 * A[0])
        expected_A = torch.matrix_exp(dA)
        expected_B = torch.linalg.solve(dA, expected_A - torch.eye(3, dtype=torch.float64)) @ (0.3 * B[0, 0])
        torch.testing.assert_close(A_bar[0, 0, 0], torch.diagonal(expected_A))
        torch.testing.assert_close(B_bar[0, 0, 0], expected_B)

    def test_mixer_is_causal(self) -> None:
        torch.manual_seed(0)
        mixer = M.MambaMixer(16, d_state=4).eval()
        x = torch.randn(2, 12, 16)
        changed = x.clone()
        changed[:, 7:] += 5.0
        with torch.no_grad():
            torch.testing.assert_close(mixer(x)[:, :7], mixer(changed)[:, :7])


class TcnTest(unittest.TestCase):
    def test_tcn_is_causal(self) -> None:
        torch.manual_seed(0)
        tcn = M.TCN(13, (32, 64), kernel_size=5, dropout=0.2).eval()
        x = torch.randn(2, 13, 20)
        changed = x.clone()
        changed[..., 11:] -= 3.0
        with torch.no_grad():
            torch.testing.assert_close(tcn(x)[..., :11], tcn(changed)[..., :11])

    def test_residual_block_uses_weight_norm(self) -> None:
        block = M.TemporalBlock(13, 32, 5, 1, 0.2)
        self.assertTrue(hasattr(block.conv1, "parametrizations"))


class TFGMambaTest(unittest.TestCase):
    def test_output_shape_and_parameter_count(self) -> None:
        torch.manual_seed(0)
        model = M.TFGMamba().eval()
        x = torch.randn(4, 20, 26)
        mask = torch.ones(4, 20, dtype=torch.bool)
        with torch.no_grad():
            self.assertEqual(model(x, mask).shape, (4,))
        n = M.count_parameters(model)
        self.assertLess(abs(n - 121_000) / 121_000, 0.10, msg=f"{n} parameters")

    def test_grsm_shapes(self) -> None:
        grsm = M.GRSM(32, 0.2).eval()
        a, c = torch.randn(2, 5, 32), torch.randn(2, 5, 32)
        self.assertEqual(grsm(a, c).shape, (2, 5, 32))

    def test_sequential_and_logdepth_models_agree(self) -> None:
        torch.manual_seed(0)
        fast = M.TFGMamba(M.TFGMambaConfig(scan="logdepth")).eval()
        ref = M.TFGMamba(M.TFGMambaConfig(scan="sequential")).eval()
        ref.load_state_dict(fast.state_dict())
        x = torch.randn(3, 17, 26)
        with torch.no_grad():
            torch.testing.assert_close(fast(x), ref(x), rtol=1e-4, atol=1e-5)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available")
    def test_mps_forward_matches_cpu(self) -> None:
        torch.manual_seed(0)
        model = M.TFGMamba().eval()
        x = torch.randn(4, 20, 26)
        with torch.no_grad():
            cpu = model(x)
            mps = model.to("mps")(x.to("mps")).cpu()
        torch.testing.assert_close(mps, cpu, rtol=1e-3, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
