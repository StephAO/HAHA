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
from tkinter import messagebox
from tkinter import simpledialog
from oai_agents.agents.agent_utils import load_agent, DummyAgent, TutorialAgent
from oai_agents.common.arguments import get_arguments
from oai_agents.common.overcooked_gui import OvercookedGUI
from itertools import product

from pylsl import StreamInfo, StreamOutlet


STEPS_PER_TRIAL = 400

ROOT = tk.Tk()
ROOT.title("Survey")
ROOT.resizable(False, False)  # This code helps to disable windows from resizing
ROOT.eval('tk::PlaceWindow . center')
# ROOT.overrideredirect(True)

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        ROOT.destroy()
        exit(0)

ROOT.protocol("WM_DELETE_WINDOW", on_closing)

info_stream = StreamInfo(name="GameData", type="Markers", channel_count=1,
                         channel_format='string', source_id='game')
outlet = StreamOutlet(info_stream)

def get_user_id_popup():
    ROOT.withdraw()
    user_id = simpledialog.askstring("User ID", "Enter User ID:")
    return 0, user_id

def run_study(args, teammates, layouts):
    trial_id, user_id = get_user_id_popup()
    agt_envt = list(product(teammates, layouts))

    # Show instructions
    show_instructions()
    # Run tutorial
    game = OvercookedGUI(args, layout_name='tutorial_0', agent='human', teammate=TutorialAgent(), horizon=1200,
                         p_idx=0, trial_id=trial_id, user_id=user_id, stream=info_stream, outlet=outlet,
                         start_message='Press Enter to Start Tutorial')
    game.on_execute()

    # Set up likert answer file
    if not os.path.exists('data/eye_tracking_data/all_likert_answers.csv'):
        with open('data/eye_tracking_data/all_likert_answers.csv', 'w+') as f:
            questions_col_headers = ','.join([f'q{i}' for i in range(len(LikertScaleGUI.questions))])
            f.write(f'trial_id,user_id,{questions_col_headers}\n')
    with open(f'data/eye_tracking_data/likert_answers_{user_id}.csv', 'w+') as f:
        questions_col_headers = ','.join([f'q{i}' for i in range(len(LikertScaleGUI.questions))])
        f.write(f'trial_id,{questions_col_headers}\n')
    try:
        # Run study and gather likert answers
        print("\n\n\nRefresh LSL streams\nSelect GameData stream\nStart LSL recording")
        lsgui = LikertScaleGUI()
        np.random.shuffle(agt_envt)
        for teammate, layout in agt_envt:
            trial_id += 1
            print(teammate, layout, flush=True)
            game = OvercookedGUI(args, layout_name=layout, agent='human', teammate=teammate, horizon=STEPS_PER_TRIAL,
                                 p_idx=0, trial_id=trial_id, user_id=user_id, stream=info_stream, outlet=outlet,
                                 start_message=f'Press Enter to Start Trial {trial_id}')
            game.on_execute()
            del game
            print('gamecomplete', flush=True)
            answers = lsgui.run()

            with open('data/eye_tracking_data/all_likert_answers.csv', 'a') as answer_file:
                answer_file.write(f'{trial_id},{user_id},{",".join([str(i) for i in answers])}\n')
            with open(f'data/eye_tracking_data/likert_answers_{user_id}.csv', 'a') as answer_file:
                answer_file.write(f'{trial_id},{",".join([str(i) for i in answers])}\n')

    except BaseException as e:
        print(e)
    trial_id += 1



def show_instructions():
    from tkinterweb import HtmlFrame
    import os

    frame = HtmlFrame(ROOT)
    # Get the current working directory
    current_dir = os.getcwd()
    # Replace backslashes with forward slashes

    current_dir = current_dir.replace("\\", "/")
    html_file_path = f'{current_dir}/scripts/eye_tracking_survey_instructions/instructions.html'
    print("HTML File Path:", html_file_path)
    frame.load_file(html_file_path, decode=None, force=False)

    def close_instructions(*args):
        ROOT.quit()
        ROOT.withdraw()
        frame.destroy()

    frame.html.config(width=1500, height=800)
    frame.pack(fill="both", expand=True)
    frame.pack_propagate(0)

    x_coordinate = ROOT.winfo_screenwidth() // 2 - 1500 // 2
    y_coordinate = ROOT.winfo_screenheight() // 2 - 800 // 2
    ROOT.geometry(f'+{x_coordinate}+{y_coordinate}')
    frame.update()


    frame.on_form_submit(close_instructions)
    ROOT.deiconify()
    ROOT.mainloop()

    

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

            self.radio_button_values.append(IntVar(self.mainframe, value=LikertScaleGUI.NON_ANSWERED_VALUE))
            rowframe = tk.Frame(self.mainframe)
            rowframe.grid(row=i, column=0, columnspan=7, pady=(2), sticky='w')
            tk_label = tk.Label(rowframe, text=f'{i + 1}. {q}', font=('Helvetica', 15))
            tk_label.grid(row=0, column=0, columnspan=7,padx=(5, 0), sticky='w')

            for j, label in enumerate(labels):
                label = f'{label:^{max_answer_length}}'
                padx = (35, 5) if j == 0 else ((5, 20) if j == (len(labels) - 1) else (5,5))
                tk_label = tk.Label(rowframe, text=label, font=('TkFixedFont', 15))
                tk_label.grid(row=1, column=j, padx=padx)
                tk_radio = tk.Radiobutton(rowframe, variable=self.radio_button_values[-1], value=j - 3)
                tk_radio.grid(row=2, column=j, padx=padx, pady=(0, 10))

        submit_button = tk.Button(self.mainframe, text="Submit", font=('TkFixedFont', 15), command=self.get_answers_and_destroy)
        submit_button.grid(row=len(LikertScaleGUI.questions), column=0, pady=(10, 10))


        self.mainframe.update()
        x_coordinate = 1920 // 2 - self.mainframe.winfo_width() // 2
        y_coordinate = 1080 // 2 - self.mainframe.winfo_height() // 2
        ROOT.geometry(f'+{x_coordinate}+{y_coordinate}')
        # self.mainframe.update()

    def get_answers_and_destroy(self):
        print('1', flush=True)
        potential_answers = [a.get() for a in self.radio_button_values]
        print('2', flush=True)
        if LikertScaleGUI.NON_ANSWERED_VALUE in potential_answers:
            pass
        else:
            self.answers = potential_answers
            print('3', flush=True)
            self.mainframe.quit()
            print('6', flush=True)
            ROOT.withdraw()
            print('4', flush=True)
            ROOT.quit()
            print('5', flush=True)


    def run(self):
        print('a', flush=True)
        for rbv in self.radio_button_values:
            print('b', flush=True)
            rbv.set(LikertScaleGUI.NON_ANSWERED_VALUE)
        print('c', flush=True)
        ROOT.deiconify()
        print('d', flush=True)
        ROOT.update()
        ROOT.mainloop()
        print('e', flush=True)
        return self.answers



if __name__ == '__main__':
    # LikertScaleGUI().run()
    args = get_arguments()
   # teammates = [load_agent(Path('agent_models/BCP'), args),  DummyAgent('random'), load_agent(Path('agent_models/SP'), args), load_agent(Path('agent_models/BCP'), args),  DummyAgent('random'), load_agent(Path('agent_models/SP'), args)]

    teammates = [load_agent(Path('agent_models/SP'), args), DummyAgent('random'),
                 load_agent(Path('agent_models/HAHA_bcp_bcp'), args)]

    # layouts = ['forced_coordination', 'counter_circuit_o_1order', 'asymmetric_advantages', 'cramped_room', 'coordination_ring']
    # layouts = ['forced_coordination', 'counter_circuit_o_1order', 'asymmetric_advantages']
    # layouts = ['asymmetric_advantages']
    layouts = ['coordination_ring', 'counter_circuit_o_1order', 'asymmetric_advantages', 'coordination_ring', 'counter_circuit_o_1order', 'asymmetric_advantages']

    # show_instructions()

    run_study(args, teammates, layouts)