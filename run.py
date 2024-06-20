import os
import sys
import time
import subprocess
from datetime import datetime

if __name__ == "__main__":
    tool = sys.argv[1]
    # tool = "mucorest"
    time_limit = "1"
    experiment_result_folder = "experiments/"
    start_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result_folder = None
    base_cov_port = 11000
    services = ["features-service", "languagetool", "ncs", "restcountries", "scs", "genome-nexus", "person-controller", "user-management", "market", "project-tracking-system"]

    for i in range(10):
        if i == 6:
            experiment_result_folder = experiment_result_folder + services[i]
            result_folder = experiment_result_folder + "/" + start_time
            os.makedirs(result_folder, exist_ok=True)
            subprocess.run("cp parameters.json " + result_folder + "/parameters.json", shell=True)

            cov_port = base_cov_port + i * 10
            print("Running " + tool + " for " + services[i] + ": " + str(cov_port))
            session = tool + '_' + services[i]
            cov_session = services[i] + "_cov"
            subprocess.run("tmux new -d -s " + cov_session + " sh get_cov.sh " + str(cov_port), shell=True)
            subprocess.run("tmux new -d -s " + session + " 'timeout " + time_limit + "h python3 run_tool.py " + tool + ' ' + services[i] + ' ' + str(cov_port) + "'", shell=True)

    time.sleep(300)
    time.sleep(int(time_limit) * 60 * 3)
    for i in range(10):
        if i == 6:
            experiment_result_folder = experiment_result_folder + services[i]
            result_folder = experiment_result_folder + "/" + start_time

            subprocess.run(
                "python collect_result.py "+result_folder, shell=True
            )





    print("Stop running services...")
    subprocess.run("sudo docker stop `sudo docker ps -a -q`", shell=True)
    time.sleep(30)
    subprocess.run("sudo docker rm `sudo docker ps -a -q`", shell=True)
    for i in range(10):
        if  i==6 :
            subprocess.run("tmux kill-sess -t " + services[i], shell=True)
            subprocess.run("tmux kill-sess -t " + services[i] + "_cov", shell=True)
            subprocess.run("tmux kill-sess -t " + tool + '_' + services[i], shell=True)
