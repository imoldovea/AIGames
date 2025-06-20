import subprocess
import sys
import unittest

from eye import main as eye_main  # assuming eye/main.py defines a main() function


class TestMain(unittest.TestCase):
    def test_main_callable(self):
        # Verify that the main function exists and is callable
        self.assertTrue(callable(eye_main), "eye.main.main should be callable")

    def test_main_runs_without_exception(self):
        # Calling main() directly should not raise an exception
        try:
            result = eye_main()
        except Exception as e:
            self.fail(f"Calling eye.main.main() raised an exception: {e}")

    def test_entry_point_execution(self):
        # Running the package as a script should exit with status 0
        completed = subprocess.run(
            [sys.executable, "-m", "eye"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(
            completed.returncode, 0,
            f"Running 'python -m eye' exited with {completed.returncode}\n"
            f"stdout:\n{completed.stdout.decode()}\n"
            f"stderr:\n{completed.stderr.decode()}"
        )


if __name__ == "__main__":
    unittest.main()
