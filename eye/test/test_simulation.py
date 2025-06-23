import unittest

from eye.config import (
    get_ga_config,
    get_mutation_config,
    get_physics_config,
    get_display_config
)


class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.ga_cfg = get_ga_config()
        self.mut_cfg = get_mutation_config()
        self.phys_cfg = get_physics_config()
        self.disp_cfg = get_display_config()
        # Initialize Simulation with default configs
        self.sim = Simulation(
            ga_config=self.ga_cfg,
            mutation_config=self.mut_cfg,
            physics_config=self.phys_cfg,
            display_config=self.disp_cfg
        )

    def test_initial_state(self):
        # Simulation should start at generation 0
        self.assertEqual(self.sim.current_generation, 0)
        # Population should be created with the correct size
        pop = getattr(self.sim, 'population', None)
        self.assertIsInstance(pop, list)
        self.assertEqual(len(pop), self.ga_cfg['population_size'])
        # Members of population should be individual objects with placeholder fitness
        for individual in pop:
            self.assertTrue(hasattr(individual, 'fitness'))
            self.assertIsNone(individual.fitness)

    def test_step_advances_generation(self):
        # Perform one simulation step or generation
        self.sim.step()
        self.assertEqual(self.sim.current_generation, 1)
        # After stepping, fitness should be evaluated
        for individual in self.sim.population:
            self.assertIsNotNone(individual.fitness)
            self.assertIsInstance(individual.fitness, (int, float))

    def test_run_multiple_generations(self):
        # Run the simulation for a few generations
        n = 5
        self.sim.run(n)
        self.assertEqual(self.sim.current_generation, n)
        # Ensure population remains correctly sized
        self.assertEqual(len(self.sim.population), self.ga_cfg['population_size'])

    def test_reset(self):
        # After running, reset should bring generation count back to zero
        self.sim.run(3)
        self.sim.reset()
        self.assertEqual(self.sim.current_generation, 0)
        # Population should be reinitialized
        self.assertEqual(len(self.sim.population), self.ga_cfg['population_size'])
        for individual in self.sim.population:
            self.assertIsNone(individual.fitness)


if __name__ == '__main__':
    unittest.main()
