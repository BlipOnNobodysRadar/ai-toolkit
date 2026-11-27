"""Training-independent contract; runnable without torch or model weights."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("distill_recipe", Path(__file__).resolve().parents[1] /
                                              "extensions_built_in/moss_distill/recipe.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class TestDistill(unittest.TestCase):
    def setUp(self):
        self.recipe = dict(model="personal.safetensors", prompts=["one", "two"], copies=4, seed=7,
                           cfg=3, width=512, height=512, steps=2, teacher_steps=5, shift=5,
                           sampler="euler", schedule="moss")
        self.rows = r.records(self.recipe)
        self.ids = [row["id"] for row in self.rows]

    def test_recipe(self):
        r.validate(self.recipe)
        self.assertEqual([x["seed"] for x in self.rows], list(range(7,15)))
        for delta in (dict(width=513), dict(copies=0), dict(steps=6), dict(sampler="unknown"),
                      dict(schedule="explicit", sigmas=[1,.5,.1]), dict(shift=float("nan")),
                      dict(teacher_sigmas=[1,0]), dict(prompts=["one","one"])):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                r.validate({**self.recipe, **delta})

    def test_dense_output_does_not_change_teacher_endpoint(self):
        calls = []
        def velocity(x,s):
            calls.append(s)
            return x * s
        # Euler endpoints: 2 -> 1 -> 0.75. Sample interior of both teacher intervals.
        path = list(r.teacher_anchors(2., [1,.5,0], [1,.9,.7,.5,.25,0], velocity))
        self.assertEqual(calls, [1,.5])
        self.assertEqual(path, [2.,1.8,1.4,1.,.875,.75])

    def test_target_solver_and_clean_endpoint(self):
        calls = []
        def v(x,s):
            calls.append(s)
            return x
        self.assertEqual(r.solver_step(v,2,1,.5,"euler"), 1)
        self.assertEqual(r.solver_step(v,2,1,.5,"heun"), 1.25)
        self.assertEqual(r.solver_step(v,2,.5,0,"heun"), 1)
        self.assertEqual(calls, [1,1,.5,.5])

    def test_rank_extremes_in_connected_groups_only(self):
        a,b,c,d,e,f,g,h = self.ids
        votes = [(a,b,'a'),(b,c,'a'),(c,d,'a'),(e,f,'tie')]
        pairs = r.ranked_pairs(self.rows,votes)
        self.assertIn((a,d), pairs)
        self.assertTrue(all(x in {a,b,c,d} for pair in pairs for x in pair))
        self.assertNotIn((a,d), r.ranked_pairs(self.rows,votes,[a]))
        # A rejected regression sample can still serve as a DPO negative.
        self.assertIn((a,d), r.ranked_pairs(self.rows,votes,[d]))
        self.assertEqual([], r.ranked_pairs(self.rows, []))
        self.assertEqual([(a,b),(c,d)], r.ranked_pairs(self.rows,[(a,b,'a'),(d,c,'b')]))

    def test_bad_votes(self):
        a,b,c,d,e,*_ = self.ids
        for vote in ((a,e,'a'), (a,a,'tie'), (a,b,'invalid'), (a,'missing','a')):
            with self.assertRaises(ValueError):
                r.ranked_pairs(self.rows, [vote])

    def test_cache_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(r.child(tmp,'my.run-01'),Path(tmp).resolve()/'my.run-01')
            for name in ('../x','x/y','..','C:\\a'):
                with self.assertRaises(ValueError):
                    r.child(tmp,name)


if __name__ == "__main__":
    unittest.main()
