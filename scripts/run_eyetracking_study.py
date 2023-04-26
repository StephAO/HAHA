import numpy as np
from pathlib import Path
import pathlib
import os

# Windows path
if os.name == 'nt':
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
from tkinter import *
from tkinter import ttk
from ttkthemes import ThemedTk

from scripts.run_overcooked_game import OvercookedGUI
from oai_agents.agents.agent_utils import load_agent, DummyAgent
from oai_agents.common.arguments import get_arguments

questions = ["The human-agent team worked fluently together:", "The human-agent team fluency improved over time:",
             "I was the most important team member:", "The agent was the most important team member:",
             "I trusted the agent to do the right thing:", "The agent helped me adapt to the task:",
             "I understood what the agent was trying to accomplish:",
             "The agent understood what I was trying to accomplish:",
             "The agent was intelligent:", "The agent was cooperative:"]


def read_trial_and_user_id():
    try:
        with open('data/eye_tracking_data/trial_user_ids.txt', 'r') as f:
            trial_id, user_id = [int(digit) for digit in f.readline().split(',')]
    except:
        trial_id, user_id = 0, 0
    return trial_id, user_id


def write_trial_and_user_id(trial_id, user_id):
    with open('data/eye_tracking_data/trial_user_ids.txt', 'w+') as f:
        f.write(f'{trial_id},{user_id}')


def run_study(args, teammates, layouts):
    np.random.shuffle(teammates)
    np.random.shuffle(layouts)
    trial_id, user_id = read_trial_and_user_id()
    if not os.path.exists('data/eye_tracking_data/likert_answers.csv'):
        with open('data/eye_tracking_data/likert_answers.csv', 'w+') as f:
            questions_col_headers = ','.join([f'q{i}' for i in range(len(questions))])
            f.write(f'trial_id,user_id,{questions_col_headers}\n')
    with open('data/eye_tracking_data/likert_answers.csv', 'a') as answer_file:
        for teammate in teammates:
            for layout in layouts:
                args.horizon = 25
                game = OvercookedGUI(args, layout_name=layout, agent='human', teammate=teammate, trial_id=trial_id,
                                     user_id=user_id)
                game.on_execute()
                answers = LikertScaleGUI().run()
                answer_file.write(f'{trial_id},{user_id},{",".join([str(i) for i in answers])}\n')
                trial_id += 1
    user_id += 1
    write_trial_and_user_id(trial_id, user_id)


class LikertScaleGUI(ThemedTk):
    NON_ANSWERED_VALUE = -4

    def __init__(self, theme='clearlooks'):
        super().__init__(fonts=True, themebg=True)
        self.set_theme(theme)
        self.title("Survey")
        self.resizable(False, False)  # This code helps to disable windows from resizing

        window_height = len(questions) * 89 + 50
        window_width = 872
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x_coordinate = screen_width // 2 - window_width // 2
        y_coordinate = screen_height // 2 - window_height // 2
        self.geometry(f'{window_width}x{window_height}+{x_coordinate}+{y_coordinate}')

        style = ttk.Style(self)
        style.theme_use('clearlooks')

        style = ttk.Style()
        style.configure('S1.TFrame', background='#888')
        style.configure('S1.TLabel', background='#888')
        style.configure('S1.TRadiobutton', background='#888')

        style.configure('S2.TFrame', background='#aaa')
        style.configure('S2.TLabel', background='#aaa')
        style.configure('S2.TRadiobutton', background='#aaa')

        style.configure('S3.TFrame', background='#ccc')
        style.configure('S3.TLabel', background='#ccc')
        style.configure('S3.TRadiobutton', background='#ccc')

        style.configure('S3.TButton', font=('Helvetica', 15))

        self.mainframe = ttk.Frame(self)
        self.mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        self.mainframe.columnconfigure(0, weight=1)
        self.mainframe.rowconfigure(0, weight=1)

        self.radio_button_values = []
        for i, q in enumerate(questions):
            labels = ['Strongly Disagree', 'Disagree', 'Somewhat Disagree', 'Neutral', 'Somewhat Agree', 'Agree',
                      'Stongly Agree']
            self.radio_button_values.append(IntVar())
            rowframe = ttk.Frame(self.mainframe, style=f'S1.TFrame')
            rowframe.grid(row=i, column=0, columnspan=7, pady=(2))

            ttk.Label(rowframe, text=f'{i + 1}. {q}', style=f'S1.TLabel', font=('Helvetica', 20)).grid(row=0, column=0,
                                                                                                       columnspan=7,
                                                                                                       padx=(5, 0),
                                                                                                       sticky='w')
            for j, label in enumerate(labels):
                s = ('S2' if (j) % 2 == 0 else 'S3')
                colframe = ttk.Frame(rowframe, style=f'{s}.TFrame')
                colframe.grid(row=1, column=j)
                ttk.Label(colframe, text=label, style=f'{s}.TLabel', font=('Helvetica', 15)).grid(row=1, column=0,
                                                                                                  padx=(5,
                                                                                                        5))  # , sticky="center")
                ttk.Radiobutton(colframe, variable=self.radio_button_values[-1], value=j - 3,
                                style=f'{s}.TRadiobutton').grid(row=2, column=0, padx=(10, 10), pady=(0, 10))

            self.radio_button_values[-1].set(LikertScaleGUI.NON_ANSWERED_VALUE)

        ttk.Button(self.mainframe, text="Submit", command=self.get_answers_and_destroy, style='S3.TButton').grid(
            row=len(questions), column=0, pady=(10, 0))

    def get_answers_and_destroy(self):
        potential_answers = [a.get() for a in self.radio_button_values]
        if LikertScaleGUI.NON_ANSWERED_VALUE in potential_answers:
            pass
        else:
            self.answers = potential_answers
            self.destroy()

    def run(self):
        for rbv in self.radio_button_values:
            rbv.set(LikertScaleGUI.NON_ANSWERED_VALUE)
        self.mainloop()
        return self.answers


if __name__ == '__main__':
    args = get_arguments()
    teammates = [load_agent(Path('agent_models/HAHA'), args), load_agent(Path('agent_models/SP'), args), DummyAgent('random')]
    layouts = ['forced_coordination', 'counter_circuit_o_1order', 'asymmetric_advantages', 'cramped_room', 'coordination_ring']
    run_study(args, teammates, layouts)
