import unittest
from types import SimpleNamespace
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('clean_share_curriculum', Path(__file__).parents[1] / 'toolkit/clean_share_curriculum.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
CleanShareSchedule, CurriculumCursor = module.CleanShareSchedule, module.CurriculumCursor
clean_share, scheduled_mean = module.clean_share, module.scheduled_mean

SPEC = '60:7:30:15:2'
SAMPLER = dict(logit_mean=1.09, logit_std=.95, cutoff=.95, uniform_mix=.01)

class CurriculumTests(unittest.TestCase):
    def cursor(self, **kwargs):
        return CurriculumCursor(SPEC, SAMPLER, 3, [['dataset'], None], **kwargs)

    def test_schedule(self):
        schedule = CleanShareSchedule(SPEC)
        for epoch, target in [(1,.6),(7,.6),(8,.5625),(11,.45),(15,.3),(16,.3),(17,.6),(18,.3),(19,.6)]:
            self.assertAlmostEqual(schedule.target(epoch), target)
        self.assertEqual(CleanShareSchedule('60:7:30:15:0').target(17), .3)

    def test_distribution(self):
        for epoch in range(1, 25):
            for cutoff in (1., .95):
                mu = scheduled_mean(SPEC, epoch, .95, cutoff, .01)
                self.assertAlmostEqual(clean_share(mu,.95,cutoff,.01), CleanShareSchedule(SPEC).target(epoch), places=10)
        self.assertLess(clean_share(1.09,.95,.95,.01), .12)

    def test_accumulation_crosses_epoch(self):
        cursor = self.cursor()
        batches = [SimpleNamespace() for _ in range(5)]
        cursor.begin()
        for batch in batches: cursor.stamp(batch)
        self.assertEqual([b._clean_share_epoch for b in batches], [1,1,1,2,2])
        with self.assertRaises(RuntimeError): cursor.state_dict()
        cursor.commit()
        resumed = self.cursor(saved=cursor.state_dict(), resuming=True)
        self.assertEqual(resumed.position, (1,2))
        self.assertEqual(resumed.stamp(SimpleNamespace()), 2)
        self.assertEqual(resumed.stamp(SimpleNamespace()), 3)

    def test_resume_guards(self):
        with self.assertRaises(ValueError): self.cursor(resuming=True)
        saved = self.cursor().state_dict()
        saved['contract']['batches_per_epoch'] = 4
        with self.assertRaises(ValueError): self.cursor(saved=saved)
        saved = self.cursor().state_dict(); saved['consumed'] = -1
        with self.assertRaises(ValueError): self.cursor(saved=saved)

    def test_prepare_guards_and_disabled(self):
        cfg = SimpleNamespace(logit_normal_clean_schedule=SPEC,
                              flow_timestep_kwargs=lambda _: SAMPLER)
        trainer = SimpleNamespace(train_config=cfg, data_loader=[1,2,3],
            data_loader_reg=None, accelerator=SimpleNamespace(num_processes=1),
            sd=SimpleNamespace(is_multistage=False),
            model_config=SimpleNamespace(refiner_name_or_path=None), step_num=0)
        self.assertEqual(module.prepare_curriculum(trainer, [['dataset'], None]).position, (0,0))
        for field, value in [('data_loader_reg', [1]), ('data_loader', None)]:
            old = getattr(trainer, field); setattr(trainer, field, value)
            with self.assertRaises(ValueError): module.prepare_curriculum(trainer, [['dataset'], None])
            setattr(trainer, field, old)
        trainer.accelerator.num_processes = 2
        with self.assertRaises(ValueError): module.prepare_curriculum(trainer, [['dataset'], None])
        cfg.logit_normal_clean_schedule = None
        self.assertIsNone(module.prepare_curriculum(trainer, [['dataset'], None]))

    def test_sampler_wiring_with_validation_exemption(self):
        import ast
        path = Path(__file__).parents[1] / 'jobs/process/BaseSDTrainProcess.py'
        tree = ast.parse(path.read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                    and isinstance(n.test, ast.BoolOp)
                    and 'logit_normal_clean_schedule' in ast.unparse(n.test)
                    and '_clean_share_validation' in ast.unparse(n.test))
        node.body = [n for n in node.body if not isinstance(n, ast.ImportFrom)]
        code = compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec')
        for validation in (False, True):
            values = dict(SAMPLER)
            env = dict(self=SimpleNamespace(train_config=SimpleNamespace(logit_normal_clean_schedule=SPEC)),
                       batch=SimpleNamespace(_clean_share_epoch=1, _clean_share_validation=validation),
                       flow_kwargs=values, scheduled_mean=scheduled_mean)
            exec(code, env)
            self.assertEqual(values['logit_mean'], 1.09 if validation else scheduled_mean(SPEC,1,.95,.95,.01))

    def test_config_incompatible_selection_rejected(self):
        import ast
        path = Path(__file__).parents[1] / 'toolkit/config_modules.py'
        tree = ast.parse(path.read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                    and ast.unparse(n.test) == 'self.logit_normal_clean_schedule is not None')
        node.body = [n for n in node.body if not isinstance(n, ast.ImportFrom)]
        code = compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec')
        cfg = SimpleNamespace(logit_normal_clean_schedule=SPEC, noise_scheduler='flowmatch',
            timestep_type='logit_normal', content_or_style='balanced', min_denoising_steps=0,
            max_denoising_steps=999, num_train_timesteps=1000)
        def run(kwargs=None): exec(code, dict(self=cfg, kwargs=kwargs or {}, CleanShareSchedule=CleanShareSchedule))
        run()
        for field, value in [('noise_scheduler','ddpm'), ('timestep_type','linear'),
                             ('content_or_style','content'), ('min_denoising_steps',1),
                             ('max_denoising_steps',998)]:
            old = getattr(cfg,field); setattr(cfg,field,value)
            with self.assertRaises(ValueError): run()
            setattr(cfg,field,old)
        for kwargs in [{'first_timestep_chance':.1}, {'linear_timesteps':True}, {'linear_timesteps2':True}]:
            with self.assertRaises(ValueError): run(kwargs)

    def test_invalid_specs(self):
        for spec in [None, [], '0:7:30:15:2','60:7:100:15:2','60:7:30:7:2','60:7:30:15:-1','nan:7:30:15:2']:
            with self.assertRaises(ValueError): CleanShareSchedule(spec)

if __name__ == '__main__': unittest.main()
