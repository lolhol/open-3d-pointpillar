import argparse

ARGUMENTS = {
    "visualize": {
        "bin_path",
        "vis_name",
    },
    "test": {
        "cfg_file",
        "bin_path",
        "ckpt_path",
    },
    "train": {
        "cfg_file",
    },
}

class CustomArgsParser(argparse.ArgumentParser):
    def __init__(self, args=ARGUMENTS):
        super(CustomArgsParser, self).__init__()

        for key in args.keys():
            self.add_argument(f'--{key}', action='store_true', help=f"Enable {key}")

        parsed_args, _ = self.parse_known_args()

        for key in args.keys():
            if getattr(parsed_args, key):
                self._add_sub_args(key, args[key])

    def _add_sub_args(self, key, sub_args):
        """Helper function to add sub-arguments based on the selected flag."""
        for sub_arg in sub_args:
            self.add_argument(f'--{sub_arg}', required=True, help=f'Argument for {key}_{sub_arg}')

    def parse(self):
        """Parse the arguments and return them as a dictionary."""
        return vars(self.parse_args())