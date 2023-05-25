import numpy as np
from pathlib import Path
import pathlib
import os

# Windows path
if os.name == 'nt':
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
from tkinter import *
import tkinter as tk

from scripts.run_overcooked_game import OvercookedGUI
from oai_agents.agents.agent_utils import load_agent, DummyAgent
from oai_agents.common.arguments import get_arguments

ROOT = tk.Tk()
ROOT.title("Survey")
ROOT.resizable(False, False)  # This code helps to disable windows from resizing
ROOT.eval('tk::PlaceWindow . center')


def get_trial_and_user_id():
    try:
        with open('data/eye_tracking_data/trial_user_ids.txt', 'r') as f:
            user_id = [int(digit) for digit in f.readline().split(',')]
    except:
        user_id = 0, 0
    return 0, user_id


def save_user_id(user_id):
    with open('data/eye_tracking_data/user_ids.txt', 'w+') as f:
        f.write(f'{user_id}')


def run_study(args, teammates, layouts):
    trial_id, user_id = get_trial_and_user_id()
    # Set up demographic answer file
    if not os.path.exists('data/eye_tracking_data/demographic_answers.csv'):
        with open('data/eye_tracking_data/demographic_answers.csv', 'w+') as f:
            questions_col_headers = ','.join([f'q{i}' for i in range(len(DemographicSurveyGUI.questions_and_answers) + 1)])
            f.write(f'user_id,{questions_col_headers}\n')
    # Set up likert answer file
    if not os.path.exists('data/eye_tracking_data/likert_answers.csv'):
        with open('data/eye_tracking_data/likert_answers.csv', 'w+') as f:
            questions_col_headers = ','.join([f'q{i}' for i in range(len(LikertScaleGUI.questions))])
            f.write(f'trial_id,user_id,{questions_col_headers}\n')
    try:
        # Get demographic answers
        with open('data/eye_tracking_data/demographic_answers.csv', 'a') as answer_file:
            demo_answers = DemographicSurveyGUI().run()
            answer_file.write(f'{user_id},{",".join([str(i) for i in demo_answers])}\n')
        # Run study and gather likert answers
        with open('data/eye_tracking_data/likert_answers.csv', 'a') as answer_file:
            np.random.shuffle(layouts)
            for layout in layouts:
                np.random.shuffle(teammates)
                for teammate in teammates:
                    args.horizon = 20
                    trial_id += 1
                    game = OvercookedGUI(args, layout_name=layout, agent='human', teammate=teammate, trial_id=trial_id,
                                         user_id=user_id)
                    game.on_execute()
                    answers = LikertScaleGUI().run()
                    answer_file.write(f'{trial_id},{user_id},{",".join([str(i) for i in answers])}\n')
    except BaseException as e:
        print(e)
    user_id += 1
    save_user_id(trial_id, user_id)


class LikertScaleGUI():
    NON_ANSWERED_VALUE = -4

    questions = ["The human-agent team worked fluently together:",
                        "I was the most important team member:",
                        "I trusted the agent to do the right thing:",
                        "I understood what the agent was trying to accomplish:",
                        "The agent was cooperative:"]

    def __init__(self):
        super().__init__()
        # self.root = tk.Tk()
        # self.root.title("Survey")
        # self.root.resizable(False, False)  # This code helps to disable windows from resizing
        #
        # self.root.eval('tk::PlaceWindow . center')

        self.mainframe = tk.Frame(ROOT)#self.root)
        self.mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        self.mainframe.columnconfigure(0, weight=1)
        self.mainframe.rowconfigure(0, weight=1)

        self.radio_button_values = []
        labels = ['Strongly Disagree', 'Disagree', 'Somewhat Disagree', 'Neutral', 'Somewhat Agree', 'Agree', 'Strongly Agree']
        max_answer_length = max([len(l) for l in labels])
        print(max_answer_length)
        for i, q in enumerate(LikertScaleGUI.questions):

            self.radio_button_values.append(IntVar(self.mainframe, value=DemographicSurveyGUI.NON_ANSWERED_VALUE))
            rowframe = tk.Frame(self.mainframe)
            rowframe.grid(row=i, column=0, columnspan=7, pady=(2), sticky='w')
            tk.Label(rowframe, text=f'{i + 1}. {q}', font=('Helvetica', 15)).grid(row=0, column=0, columnspan=7,
                                                                                  padx=(5, 0), sticky='w')
            for j, label in enumerate(labels):
                label = f'{label:^{max_answer_length}}'
                padx = (35, 5) if j == 0 else ((5, 20) if j == (len(labels) - 1) else (5,5))
                tk.Label(rowframe, text=label, font=('TkFixedFont', 15)).grid(row=1, column=j, padx=padx)
                tk.Radiobutton(rowframe, variable=self.radio_button_values[-1], value=j - 3).grid(row=2, column=j, padx=padx, pady=(0, 10))

        tk.Button(self.mainframe, text="Submit", font=('TkFixedFont', 15), command=self.get_answers_and_destroy)\
                 .grid(row=len(LikertScaleGUI.questions), column=0, pady=(10, 10))

        self.mainframe.update()
        x_coordinate = ROOT.winfo_screenwidth() // 2 - ROOT.winfo_width() // 2
        y_coordinate = ROOT.winfo_screenheight() // 2 - ROOT.winfo_height() // 2
        ROOT.geometry(f'+{x_coordinate}+{y_coordinate}')

    def get_answers_and_destroy(self):
        potential_answers = [a.get() for a in self.radio_button_values]
        if LikertScaleGUI.NON_ANSWERED_VALUE in potential_answers:
            pass
        else:
            self.answers = potential_answers
            ROOT.quit()
            ROOT.withdraw()
            self.mainframe.destroy()
            print(self.answers)

    def run(self):
        for rbv in self.radio_button_values:
            rbv.set(LikertScaleGUI.NON_ANSWERED_VALUE)
        ROOT.deiconify()
        ROOT.mainloop()
        return self.answers


class DemographicSurveyGUI():
    NON_ANSWERED_VALUE = -1

    questions_and_answers = [('Gender:', ['Male', 'Female', 'Non-binary', 'Prefer not to say']),
                             ('Handedness:', ['Right', 'Left', 'Ambidextrous']),
                             ('What is your highest level of education:', ['No formal Education', 'High School', 'College', 'Vocational Training', 'Bachelors', 'Masters', 'Doctorate/Phd']),
                             ('How much experience do you have playing video games:', ['Never played', '6 months', '1 year', '2-5 years', '5-10 years', '10+ years']),
                             ('During the average week how many hours do you currently spend playing video games?:', ['0 hours', '<1 hour', '1-2 hours', '3-5 hours', '5-10 hours', '10+ hours']),
                             ('How would you rate you video game ability:', ['Very low skill level', 'Low skill level', 'Moderate skill level', 'High skill level', 'Very high skill level']),
                             ('Please select your favorite game genre from the following:', ['First-person shooter', 'Driving/Racing', 'Sports', 'Real-Time Strategy', 'Role-playing (RPG)', 'Couch Co-op', 'Action/Adventure']),
                             ('Please select your 2nd favorite game genre from the following:', ['First-person shooter', 'Driving/Racing', 'Sports', 'Real-Time Strategy', 'Role-playing (RPG)', 'Couch Co-op', 'Action/Adventure']),
                             ('Please select your 3rd favorite game genre from the following:', ['First-person shooter', 'Driving/Racing', 'Sports', 'Real-Time Strategy', 'Role-playing (RPG)', 'Couch Co-op', 'Action/Adventure'])]

    def __init__(self):
        super().__init__()
        self.mainframe = tk.Frame(ROOT)
        self.mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        self.mainframe.columnconfigure(0, weight=1)
        self.mainframe.rowconfigure(0, weight=1)

        rowframe = tk.Frame(self.mainframe)
        rowframe.grid(row=0, column=0, columnspan=7, pady=(2), sticky='w')
        tk.Label(rowframe, text=f'{1}. Age (years):', font=('Helvetica', 15)).grid(row=0, column=0, columnspan=7, padx=(5, 0), sticky='w')
        self.age_value = tk.Text(rowframe, height=1, width = 5, font=('Helvetica', 15))
        self.age_value.grid(row=1, column=0, columnspan=7, padx=(32, 5), sticky='w')

        self.radio_button_values = []
        max_answer_length = max([len(a) for qas in DemographicSurveyGUI.questions_and_answers for a in qas[1]])
        for i, (question, answers) in enumerate(DemographicSurveyGUI.questions_and_answers):
            self.radio_button_values.append(IntVar(self.mainframe, value=DemographicSurveyGUI.NON_ANSWERED_VALUE))
            rowframe = tk.Frame(self.mainframe)
            rowframe.grid(row=i+1, column=0, columnspan=7, pady=(2), sticky='w')

            tk.Label(rowframe, text=f'{i + 2}. {question}', font=('Helvetica', 15)).grid(row=0, column=0, columnspan=7,
                                                                                        padx=(5, 0), sticky='w')

            for j, answer in enumerate(answers):
                answer = f'{answer:^{max_answer_length}}'
                columnspan = (1 if j < (len(answers) - 1) else 8 - len(answers))
                padx = (35, 5) if j == 0 else ((5, 20) if j == (len(answer) - 1) else (5, 5))
                tk.Label(rowframe, text=answer, font=('TkFixedFont', 15)).grid(row=1, column=j, padx=padx, columnspan=columnspan)
                tk.Radiobutton(rowframe, variable=self.radio_button_values[-1], value=j).grid(row=2, column=j, padx=padx, pady=(0, 5), columnspan=columnspan)

        tk.Button(self.mainframe, text="Submit", font=('TkFixedFont', 15), command=self.get_answers_and_destroy).grid(row=len(DemographicSurveyGUI.questions_and_answers)+1, column=0, pady=(10, 10))

        self.mainframe.update()
        x_coordinate = ROOT.winfo_screenwidth() // 2 - ROOT.winfo_width() // 2
        y_coordinate = ROOT.winfo_screenheight() // 2 - ROOT.winfo_height() // 2
        ROOT.geometry(f'+{x_coordinate}+{y_coordinate}')

    def get_answers_and_destroy(self):
        age_value = self.age_value.get(1.0, "end-1c")
        try:
            age_value = int(age_value)
        except ValueError:
            return
        potential_answers = [a.get() for a in self.radio_button_values]
        if DemographicSurveyGUI.NON_ANSWERED_VALUE in potential_answers:
            pass
        else:
            self.answers = [age_value] + potential_answers
            # self.answers = [age_value] + [DemographicSurveyGUI.questions_and_answers[i][1][a] for i, a in enumerate(potential_answers)]
            ROOT.quit()
            ROOT.withdraw()
            self.mainframe.destroy()
            print(self.answers)

    def run(self):
        for rbv in self.radio_button_values:
            rbv.set(DemographicSurveyGUI.NON_ANSWERED_VALUE)
        ROOT.deiconify()
        ROOT.mainloop()
        return self.answers


if __name__ == '__main__':
    # LikertScaleGUI().run()
    # DemographicSurveyGUI().run()
    args = get_arguments()
    teammates = [load_agent(Path('agent_models/HAHA'), args),  DummyAgent('random')]
    layouts = ['forced_coordination', 'counter_circuit_o_1order', 'asymmetric_advantages', 'cramped_room', 'coordination_ring']
    run_study(args, teammates, layouts)
