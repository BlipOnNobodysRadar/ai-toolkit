"""toolkit/lora_scalar_export.py: folding the learned scalar into the up factor for loaders that ignore ``.scalar``
(ComfyUI). CPU only. The last test runs the trainer's own MOSS save conversion to pin the key layout it folds."""
import os
import tempfile
import unittest

import torch

from toolkit.lora_scalar_export import export_file, fold_scalars


def _delta(sd, prefix, scalar=1.0):
    return (sd[f"{prefix}.lora_B.weight"].float() @ sd[f"{prefix}.lora_A.weight"].float()) * scalar


class FoldScalarsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_plain_module_delta_is_kept_and_key_dropped(self):
        sd = {
            "m.lora_A.weight": torch.randn(4, 8),
            "m.lora_B.weight": torch.randn(6, 4),
            "m.scalar": torch.tensor(0.137),
            "n.lora_A.weight": torch.randn(4, 8),
            "n.lora_B.weight": torch.randn(6, 4),
        }
        out, folded = fold_scalars(sd)
        self.assertNotIn("m.scalar", out)
        self.assertEqual(list(folded), ["m.scalar"])
        torch.testing.assert_close(_delta(out, "m"), _delta(sd, "m", 0.137), rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.equal(out["m.lora_A.weight"], sd["m.lora_A.weight"]))
        self.assertTrue(torch.equal(out["n.lora_B.weight"], sd["n.lora_B.weight"]))   # no scalar: untouched

    def test_lora_up_names_and_dtype(self):
        sd = {
            "m.lora_down.weight": torch.randn(4, 8).to(torch.bfloat16),
            "m.lora_up.weight": torch.randn(6, 4).to(torch.bfloat16),
            "m.scalar": torch.tensor(-0.25, dtype=torch.bfloat16),
        }
        out, _ = fold_scalars(sd)
        self.assertEqual(out["m.lora_up.weight"].dtype, torch.bfloat16)
        torch.testing.assert_close(out["m.lora_up.weight"].float(), sd["m.lora_up.weight"].float() * -0.25)

    def test_fused_qkv_scalar_goes_into_q_k_and_v(self):
        sd = {"b.proj.qkv.scalar": torch.tensor(0.5)}
        for p in "qkv":
            sd[f"b.proj.{p}.lora_A.weight"] = torch.randn(4, 8)
            sd[f"b.proj.{p}.lora_B.weight"] = torch.randn(6, 4)
        out, folded = fold_scalars(sd)
        self.assertEqual(len(folded), 1)
        self.assertFalse(any(k.endswith(".scalar") for k in out))
        for p in "qkv":
            torch.testing.assert_close(_delta(out, f"b.proj.{p}"), _delta(sd, f"b.proj.{p}", 0.5))

    def test_scalar_without_an_up_factor_is_an_error(self):
        with self.assertRaises(KeyError):
            fold_scalars({"m.lora_A.weight": torch.randn(4, 8), "m.scalar": torch.tensor(0.1)})
        with self.assertRaises(ValueError):
            fold_scalars({"m.lora_B.weight": torch.randn(6, 4), "m.scalar": torch.tensor(float("nan"))})

    def test_export_file_writes_a_copy_and_never_the_source(self):
        from safetensors import safe_open
        from safetensors.torch import save_file
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "run.safetensors")
            save_file({"m.lora_A.weight": torch.randn(4, 8), "m.lora_B.weight": torch.randn(6, 4),
                       "m.scalar": torch.tensor(0.2)}, src, {"name": "run"})
            dst, n, lo, hi = export_file(src)
            self.assertEqual((os.path.basename(dst), n), ("run-comfy.safetensors", 1))
            self.assertAlmostEqual(lo, 0.2, places=6)
            with safe_open(dst, framework="pt") as f:
                self.assertEqual(sorted(f.keys()), ["m.lora_A.weight", "m.lora_B.weight"])
                self.assertEqual(f.metadata()["scalar_folded"], "1")
                self.assertEqual(f.metadata()["name"], "run")
            with safe_open(src, framework="pt") as f:
                self.assertIn("m.scalar", f.keys())
            with self.assertRaises(ValueError):
                export_file(src, src)

    def test_matches_the_layout_of_the_moss_save_conversion(self):
        """The trainer saves a fused qkv module as q/k/v with a shared down factor and leaves the scalar on the fused
        name; the fold of that layout equals the fused module's own delta times its scalar."""
        from extensions_built_in.diffusion_models.moss_image.moss_image import MossImageModel
        fused = {
            "diffusion_model.layers.0.proj.qkv.lora_A.weight": torch.randn(4, 8),
            "diffusion_model.layers.0.proj.qkv.lora_B.weight": torch.randn(18, 4),
            "diffusion_model.layers.0.proj.qkv.scalar": torch.tensor(0.3),
            "diffusion_model.layers.0.proj.o.lora_A.weight": torch.randn(4, 8),
            "diffusion_model.layers.0.proj.o.lora_B.weight": torch.randn(6, 4),
            "diffusion_model.layers.0.proj.o.scalar": torch.tensor(0.7),
        }
        saved = {}
        for key, value in fused.items():   # the split step of convert_lora_weights_before_save, without a model object
            base, sep, suffix = key.partition(".proj.qkv.")
            if not sep:
                saved[key] = value
            elif suffix in MossImageModel._QKV_SPLIT_ROWS:
                for name, part in zip("qkv", value.chunk(3, dim=0)):
                    saved[f"{base}.proj.{name}.{suffix}"] = part.contiguous()
            elif suffix in MossImageModel._QKV_SHARED:
                for name in "qkv":
                    saved[f"{base}.proj.{name}.{suffix}"] = value.clone()
            else:
                saved[key] = value
        self.assertIn("diffusion_model.layers.0.proj.qkv.scalar", saved)   # the layout this tool exists for
        out, folded = fold_scalars(saved)
        self.assertEqual(len(folded), 2)
        want = _delta(fused, "diffusion_model.layers.0.proj.qkv", 0.3)
        got = torch.cat([_delta(out, f"diffusion_model.layers.0.proj.{p}") for p in "qkv"], dim=0)
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(_delta(out, "diffusion_model.layers.0.proj.o"),
                                   _delta(fused, "diffusion_model.layers.0.proj.o", 0.7), rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
