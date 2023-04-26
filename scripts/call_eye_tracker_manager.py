import subprocess
import platform
import glob
import os
import tobii_research as tr

found_eyetrackers = tr.find_all_eyetrackers()
my_eyetracker = found_eyetrackers[0]
print("Address: " + my_eyetracker.address)
print("Model: " + my_eyetracker.model)
print("Name (It's OK if this is empty): " + my_eyetracker.device_name)
print("Serial number: " + my_eyetracker.serial_number)


def call_eyetracker_manager_example():
    try:
        os_type = platform.system()
        print(os_type)
        ETM_PATH = ''
        DEVICE_ADDRESS = ''
        if os_type == "Windows":
            ETM_PATH = "C:/Users/ARL/AppData/Local/Programs/TobiiProEyeTrackerManager/TobiiProEyeTrackerManager.exe"
            DEVICE_ADDRESS = "tobii-ttp://IS404-100107844652"
        elif os_type == "Linux":
            ETM_PATH = "TobiiProEyeTrackerManager"
            DEVICE_ADDRESS = "tobii-ttp://TOBII-IS404-100107417574"
        elif os_type == "Darwin":
            ETM_PATH = "/Applications/TobiiProEyeTrackerManager.app/Contents/MacOS/TobiiProEyeTrackerManager"
            DEVICE_ADDRESS = "tobii-ttp://TOBII-IS404-100107417574"
        else:
            print("Unsupported...")
            exit(1)

        eyetracker = tr.EyeTracker(DEVICE_ADDRESS)

        mode = "usercalibration"

        print("address " + eyetracker.address)
        etm_p = subprocess.Popen([ETM_PATH,
                                "--device-address=" + eyetracker.address,
                                "--mode=" + mode],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                shell=False)

        stdout, stderr = etm_p.communicate()  # Returns a tuple with (stdout, stderr)

        if etm_p.returncode == 0:
                print("Eye Tracker Manager was called successfully!")
        else:
                print("Eye Tracker Manager call returned the error code: " + str(etm_p.returncode))
                errlog = None
                if os_type == "Windows":
                    errlog = stdout  # On Windows ETM error messages are logged to stdout
                else:
                    errlog = stderr

                for line in errlog.splitlines():
                    if line.startswith("ETM Error:"):
                        print(line)

    except Exception as e:
        print(e)

if __name__ == "__main__":
    call_eyetracker_manager_example()