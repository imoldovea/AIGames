import unittest

from eye.config import (
    get_ga_config,
    get_mutation_config,
    get_physics_config,
    get_display_config
)


class TestConfig(unittest.TestCase):
    def test_ga_config(self):
        config = get_ga_config()
        # Ensure it's a dictionary
        self.assertIsInstance(config, dict)
        # Expected keys in GA config
        expected_keys = {
            'population_size',
            'generations',
            'selection_rate',
            'crossover_rate',
            'mutation_rate'
        }
        self.assertEqual(set(config.keys()), expected_keys)
        # Check value types and reasonable ranges
        self.assertIsInstance(config['population_size'], int)
        self.assertGreater(config['population_size'], 0)
        for key in ('generations',):
            self.assertIsInstance(config[key], int)
            self.assertGreaterEqual(config[key], 1)
        for key in ('selection_rate', 'crossover_rate', 'mutation_rate'):
            self.assertIsInstance(config[key], float)
            self.assertGreaterEqual(config[key], 0.0)
            self.assertLessEqual(config[key], 1.0)

    def test_mutation_config(self):
        config = get_mutation_config()
        self.assertIsInstance(config, dict)
        expected_keys = {
            'mutation_type',
            'mutation_strength',
            'mutation_rate'
        }
        self.assertEqual(set(config.keys()), expected_keys)
        # mutation_type should be a string
        self.assertIsInstance(config['mutation_type'], str)
        # rates and strengths are floats within expected bounds
        self.assertIsInstance(config['mutation_rate'], float)
        self.assertGreaterEqual(config['mutation_rate'], 0.0)
        self.assertLessEqual(config['mutation_rate'], 1.0)
        self.assertIsInstance(config['mutation_strength'], float)
        self.assertGreaterEqual(config['mutation_strength'], 0.0)

    def test_physics_config(self):
        config = get_physics_config()
        self.assertIsInstance(config, dict)
        expected_keys = {
            'gravity',
            'friction',
            'air_resistance',
            'time_step'
        }
        self.assertEqual(set(config.keys()), expected_keys)
        # All physical parameters should be numeric
        for key in expected_keys:
            self.assertIsInstance(config[key], (int, float))
        # gravity is usually positive
        self.assertGreater(config['gravity'], 0)
        # time_step is a small positive float
        self.assertGreater(config['time_step'], 0.0)

    def test_display_config(self):
        config = get_display_config()
        self.assertIsInstance(config, dict)
        expected_keys = {
            'width',
            'height',
            'fullscreen',
            'bg_color'
        }
        self.assertEqual(set(config.keys()), expected_keys)
        # width and height as positive ints
        self.assertIsInstance(config['width'], int)
        self.assertGreater(config['width'], 0)
        self.assertIsInstance(config['height'], int)
        self.assertGreater(config['height'], 0)
        # fullscreen is boolean
        self.assertIsInstance(config['fullscreen'], bool)
        # bg_color is a string, e.g., hex or named
        self.assertIsInstance(config['bg_color'], str)


if __name__ == '__main__':
    unittest.main()
