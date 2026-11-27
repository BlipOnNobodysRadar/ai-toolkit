from toolkit.extension import Extension


class MossDistillExtension(Extension):
    uid = "moss_step_distill"
    name = "MOSS step distillation"

    @classmethod
    def get_process(cls):
        from .process import MossDistill
        return MossDistill


AI_TOOLKIT_EXTENSIONS = [MossDistillExtension]
