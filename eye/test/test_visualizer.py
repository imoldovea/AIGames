import unittest

import numpy as np
from PIL import Image

from eye.config import get_display_config


class DummyIndividual:
    """Minimal dummy individual with position and fitness attributes."""

    def __init__(self, x, y, fitness=None):
        self.x = x
        self.y = y
        self.fitness = fitness


class TestVisualizer(unittest.TestCase):
    def setUp(self):
        self.display_cfg = get_display_config()
        # Instantiate the Visualizer with the display configuration
        self.viz = Visualizer(self.display_cfg)

    def test_instantiation(self):
        # The visualizer should store the display configuration
        self.assertTrue(hasattr(self.viz, 'display_config'))
        self.assertEqual(self.viz.display_config, self.display_cfg)

    def test_render_with_population(self):
        # Create a dummy population of 10 individuals
        pop = [DummyIndividual(x=i * 10, y=i * 5, fitness=i) for i in range(10)]
        output = self.viz.render(pop)

        # The render output should be either a numpy array or PIL Image
        self.assertTrue(isinstance(output, (np.ndarray, Image.Image)))

        # If numpy array, check dimensions and dtype
        if isinstance(output, np.ndarray):
            height, width = output.shape[:2]
            self.assertEqual((width, height),
                             (self.display_cfg['width'], self.display_cfg['height']))
            # Expect 3 color channels (RGB)
            self.assertEqual(output.shape[2], 3)
            self.assertIn(output.dtype, (np.uint8, np.float32, np.float64))
        else:
            # PIL Image: size is (width, height) and mode is RGB
            self.assertEqual(output.size,
                             (self.display_cfg['width'], self.display_cfg['height']))
            self.assertEqual(output.mode, 'RGB')

    def test_render_empty_population(self):
        # Rendering an empty population should still produce a valid image
        output = self.viz.render([])
        self.assertTrue(isinstance(output, (np.ndarray, Image.Image)))

    def test_close_method(self):
        # If the visualizer opens external resources, it should provide a close method
        if hasattr(self.viz, 'close'):
            # Calling close should not raise
            try:
                self.viz.close()
            except Exception as e:
                self.fail(f"Visualizer.close() raised an exception: {e}")

    def tearDown(self):
        # Ensure resources are cleaned up
        if hasattr(self.viz, 'close'):
            self.viz.close()


if __name__ == '__main__':
    unittest.main()
