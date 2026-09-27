import subprocess
import time
import os

test_exp = [
    ("crossq_pp",600000),
]


experiments = test_exp


root_dir = './results'
os.environ["CARLA_ROOT"] = "/home/user/yechan_work/carla/CARLA_0.9.15"
os.environ["SDL_VIDEODRIVER"] = "dummy"

towns = ["town01"]

def kill_carla_server():
    print("Killing Carla server\n")
    time.sleep(1)
    subprocess.run(["killall", "-9", "CarlaUE4-Linux-Shipping"])
    time.sleep(4)


def get_last_model_path(id_config):
    dirs = os.listdir(root_dir)
    temp_path = None
    for dir in sorted(dirs, reverse=True):
        id_dir = dir.split('_')[-1][2:]
        if id_config == id_dir:
            temp_path = os.path.join(root_dir, dir)
            break
    if temp_path is None:
        raise Exception('Model not found')
    dirs = os.listdir(temp_path)


    max_steps = 0
    latest_model = ''
    for file in dirs:
        if file.endswith('.zip') and file.startswith('model_'):
            steps = int(file.split('_')[1].split('.')[0])
            if steps > max_steps:
                max_steps = steps
                latest_model = file

    return os.path.join(temp_path, latest_model)


def main():
    for config, steps in experiments:
        kill_carla_server()

        for town in towns:
            kill_carla_server()

            for folder in os.listdir(root_dir):
                if "id" in folder and folder.split("id")[1] == config:
                    continue

            print(f"Running experiment {config} with {steps} steps\n")
            args_train = [
                "--config", config,
                "--total_timesteps", str(steps),
                "--save_dir", str(root_dir),
                "--town", town,
                "--no_render"
            ]
            subprocess.run(["python", "train.py"] + args_train)
            time.sleep(5)
            kill_carla_server()


if __name__ == "__main__":
    main()
