import importlib.util
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/install_skill.py'
spec = importlib.util.spec_from_file_location('install_skill', SOURCE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class InstallTests(unittest.TestCase):
    def test_installs_complete_skill_without_machine_record(self):
        with tempfile.TemporaryDirectory() as directory:
            target = module.install_skill(Path(directory) / 'skills with spaces')
            self.assertTrue((target / 'SKILL.md').is_file())
            self.assertTrue((target / 'scripts/nsight_profile.py').is_file())
            self.assertTrue((target / 'references/native-cpu.md').is_file())
            self.assertFalse((target / 'references/local-install.json').exists())

    def test_existing_installation_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / module.NAME
            target.mkdir()
            (target / 'local-notes.txt').write_text('preserve this installation')
            with self.assertRaises(FileExistsError):
                module.install_skill(directory)
            self.assertEqual((target / 'local-notes.txt').read_text(), 'preserve this installation')
            self.assertEqual([p.name for p in target.iterdir()], ['local-notes.txt'])


if __name__ == '__main__':
    unittest.main()
