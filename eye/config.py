# eye/config.py
import configparser
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.ini')

config = configparser.ConfigParser()
config.read(CONFIG_PATH)


def get_ga_config():
    return {
        'population_size': config.getint('GA', 'population_size'),
        'generations': config.getint('GA', 'generations'),
        'fitness_threshold': config.getfloat('GA', 'fitness_threshold'),
        'grid_size': (config.getint('GA', 'grid_rows'), config.getint('GA', 'grid_cols'))
    }


def get_mutation_config():
    return {
        'initial': {
            'fluid': config.getfloat('Mutation', 'initial_fluid'),
            'lens': config.getfloat('Mutation', 'initial_lens'),
            'retina': config.getfloat('Mutation', 'initial_retina')
        },
        'priority': {
            'fluid': config.getfloat('Mutation', 'priority_fluid'),
            'lens': config.getfloat('Mutation', 'priority_lens'),
            'retina': config.getfloat('Mutation', 'priority_retina')
        }
    }


def get_physics_config():
    return {
        'refractive_index': {
            'fluid': config.getfloat('Physics', 'refractive_index_fluid'),
            'lens': config.getfloat('Physics', 'refractive_index_lens')
        }
    }


def get_display_config():
    return {
        'cell_size_px': config.getint('Display', 'cell_size_px'),
        'fps': config.getint('Display', 'fps')
    }


if __name__ == "__main__":
    _test_config()
