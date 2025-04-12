import subprocess

OUTPUT = "output/"
if is_port_in_use(6006):
    logging.warning("TensorBoard is already running on port 6006. Skipping startup.")
else:
    tensorboard_process = subprocess.Popen(
        ["tensorboard", "--logdir", f"{OUTPUT}tensorboard_data", "--port", "6006"])