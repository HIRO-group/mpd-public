import yaml
import pdb


def load_params_from_yaml(path: str):
    with open(path, "r") as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def get_start_goal_from_yaml(path: str):
    with open(path, "r") as stream:
            try:
                data = yaml.load(
                    stream, Loader=yaml.Loader
                )  # DO NOT USE UNTRUSTED YAMLS
            except yaml.YAMLError as exc:
                print(exc)
    start_pos = data['start_state']['joint_state']['position'][0:7]
    goal_pos = []
    for joint in data['goal_constraints'][0]['joint_constraints']:
        goal_pos.append(joint['position'])
    return start_pos, goal_pos
