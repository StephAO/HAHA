import numpy as np
from pathlib import Path
import pathlib
from os import name

# Windows path
if name == 'nt':
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
from tkinter import *

from scripts.run_overcooked_game import OvercookedGUI
from oai_agents.agents.agent_utils import load_agent, DummyAgent
from oai_agents.common.arguments import get_arguments

questions = ["The human-agent team worked fluently together:", "The human-agent team fluency improved over time:",
            "I was the most important team member:", "The agent was the most important team member:",
            "I trusted the agent to do the right thing:", "The agent helped me adapt to the task:",
            "I understood what the agent was trying to accomplish:", "The agent understood what I was trying to accomplish:",
            "The agent was intelligent:", "The agent was cooperative:"]


def read_trial_and_user_id():
    return 0, 1

def write_trial_and_user_id(trial_id, user_id):
    return 0, 1


def run_study(args, teammates, layouts):
    np.random.shuffle(teammates)
    np.random.shuffle(layouts)
    trial_id, user_id = read_trial_and_user_id()
    for teammate in teammates:
        for layout in layouts:
            game = OvercookedGUI(args, layout_name=layout, agent='human', teammate=teammate, trial_id=trial_id, user_id=user_id)
            game.on_execute()
            run_likert_scale()
            trial_id += 1

    write_trial_and_user_id(trial_id, user_id)


def run_likert_scale():
    NON_ANSWERED_VALUE = -4
    root = Tk()
    root.title("Centering windows")
    root.resizable(False, False)  # This code helps to disable windows from resizing

    window_height = 1000
    window_width = 1500

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    x_coordinate = screen_width // 2 - window_width // 2
    y_coordinate = screen_height // 2 - window_height // 2

    root.geometry("{}x{}+{}+{}".format(window_width, window_height, x_coordinate, y_coordinate))

    answers = []
    for i, q in enumerate(questions):
        labels = ['Strongly Disagree', 'Disagree', 'Somewhat Disagree', 'Neutral', 'Somewhat Agree', 'Agree',
                  'Stongly Agree']
        answers.append(IntVar())
        q_text = Label(root, text=f'{i+1}.: q', font=("Arial", 25))
        q_text.grid(row=i * 2, column=0, columnspan=5, pady=(15, 5), padx=(10, 0), sticky="w")
        for j, label in enumerate(labels):
            radio_button = Radiobutton(root, text=label, variable=answers[-1], value=j - 3, font=("Arial", 15), width=17)
            radio_button.grid(row=i * 2 + 1, column=j, padx=(0, 0), sticky="w")
        answers[-1].set(NON_ANSWERED_VALUE)

    def get_answers_and_destroy():
        nonlocal answers
        potential_answers = [a.get() for a in answers]
        if NON_ANSWERED_VALUE in potential_answers:
            pass
        else:
            answers = potential_answers
            root.destroy()

    submit_button = Button(root, text="Submit", width=10, command=get_answers_and_destroy, font=("Arial", 20))
    submit_button.grid(row=len(questions)*2+1, column=3, pady=(25, 0))

    # root.grid_columnconfigure((0, 7), weight=1)

    root.mainloop()
    print(answers)
    return answers


if __name__ == '__main__':
    args = get_arguments()
    teammates = [load_agent(Path('agent_models/HAHA'), args), load_agent(Path('agent_models/SP'), args),
                 DummyAgent('random')]
    layouts = ['forced_coordination', 'counter_circuit_o_1order', 'asymmetric_advantages', 'cramped_room',
               'coordination_ring']
    run_study(args, teammates, layouts)
