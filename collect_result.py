import os
import re
import json
import subprocess
import sys
import time
from collections import Counter
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

try:
    from BeautifulSoup import BeautifulSoup
except ImportError:
    from bs4 import BeautifulSoup

def doesExistCoverageReport(filepath):
    subdirs = [x[0] for x in os.walk(filepath)]
    for subdir in subdirs:
        if "/target/site/jacoco" in subdir:
            return True

def count_coverage(path, port):
    class_files = []
    source_files = []
    report_dest_dir = ""

    jacoco_command2 = ''
    subdirs = [x[0] for x in os.walk(path)]
    for subdir in subdirs:
        if '/target/classes/' in subdir:
            target_dir = subdir[:subdir.rfind('/target/classes/') + 15]
            if report_dest_dir == "":
                report_dest_dir = subdir[:subdir.rfind('/target/classes/') + 7]
            if target_dir not in class_files:
                class_files.append(target_dir)
                jacoco_command2 = jacoco_command2 + ' --classfiles ' + target_dir
        elif '/build/classes/' in subdir:
            target_dir = subdir[:subdir.rfind('/build/classes/') + 14]
            if report_dest_dir == "":
                report_dest_dir = subdir[:subdir.rfind('/build/classes/') + 6]
            if target_dir not in class_files:
                class_files.append(target_dir)
                jacoco_command2 = jacoco_command2 + ' --classfiles ' + target_dir
        if '/src/main/java/' in subdir:
            source_dir = subdir[:subdir.rfind('/src/main/java/') + len('/src/main/java/') - 1]
            if source_dir not in source_files:
                source_files.append(source_dir)
                jacoco_command2 = jacoco_command2 + ' --sourcefiles ' + source_dir

    jacoco_command2 = jacoco_command2 + ' --csv '
    jacoco_command1 = 'java -jar org.jacoco.cli-0.8.7-nodeps.jar report '
    jacoco_file = port + '.csv'
    jacoco_command2 += jacoco_file
    jacoco_command2 += ' --html ' + report_dest_dir + "/site/jacoco"
    print(jacoco_command1 + "jacoco" + port + ".exec" + jacoco_command2)
    subprocess.run(jacoco_command1 + "jacoco" + port + ".exec" + jacoco_command2, shell=True)


def get_list_of_HTML_report(file_path):
    def find_files(directory, pattern):
        matches = []
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.endswith(pattern):
                    matches.append(os.path.join(root, filename))
        return matches

    start_time = time.time()
    file_paths = []
    # file_paths = find_files(file_path, ".java.html")

    target_sub_dirs = []
    subdirs = [x[0] for x in os.walk(file_path)]

    for subdir in subdirs:
        if '/target/site/jacoco/' in subdir:
            target_sub_dir = subdir[:subdir.rfind('/target/site/jacoco/') + len('/target/site/jacoco/') - 1]
            if target_sub_dir not in target_sub_dirs:
                target_sub_dirs.append(target_sub_dir)
    for target_sub_dir in target_sub_dirs:
        matches = find_files(target_sub_dir, ".java.html")
        file_paths.extend(matches)

    end_time = time.time()
    exe_time = end_time - start_time
    print(exe_time)
    return file_paths
def analyse_HTML_report(path,final_report):


    with open(path, "r") as file:
        html_content = file.read()

        soup = BeautifulSoup(html_content, "html.parser")  # Parse the HTML

        spans_with_fc = soup.find_all("span", class_="fc")  # Find all matching spans
        fc_ids = [  path+"_"+span.get("id") for span in spans_with_fc]  # Extract IDs

        spans_with_nc = soup.find_all("span", class_="nc")  # Find all matching spans
        nc_ids = [path+"_"+span.get("id") for span in spans_with_nc]

        spans_with_pc = soup.find_all("span", class_="pc")  # Find all matching spans
        pc_ids = [path+"_"+span.get("id") for span in spans_with_pc]

        final_report["fc"].extend(fc_ids)
        final_report["nc"].extend(nc_ids)
        final_report["pc"].extend(pc_ids)
    return


def get_line_cov_analysis(path):
    start_time=time.time()
    analysis_result = {}
    html_reports_paths = get_list_of_HTML_report(path)
    final_analysis_result = {
        "fc": [],
        "nc": [],
        "pc": [],
        "bfc": [],
        "bnc": []
    }
    for path in html_reports_paths:
        analyse_HTML_report(path,final_report=final_analysis_result)
        # analysis_result[path] = analysis


    # for file_path in analysis_result:
    #     for line in analysis_result[file_path]['fc']:
    #         final_analysis_result['fc'].append(file_path + "_" + line)
    #     for line in analysis_result[file_path]['nc']:
    #         final_analysis_result['nc'].append(file_path + "_" + line)
    #     for line in analysis_result[file_path]['pc']:
    #         final_analysis_result['pc'].append(file_path + "_" + line)
    endtime=time.time()
    exe_time = endtime-start_time
    print(exe_time)

    return final_analysis_result

def parse_log_file(file_path):
    log_data = []
    status2xx = 0
    status4xx = 0
    status5xx = 0
    request_index = 1
    with open(file_path, 'r') as f:
        current_log = {}
        for line in f:
            if "========REQUEST========" in line:
                current_log = {'request': {}, 'response': {}}
            elif "========RESPONSE========" in line:
                if "response" in current_log:
                    current_log['response']['timestamp'] = float(f.readline().strip())
                    current_log['response']['status_code'] = int(f.readline().strip())
                    current_log['index'] = request_index
                    current_log['response']['has_read'] = True
                    request_index += 1
                    status = current_log['response']['status_code'] // 100
                    if status == 2:
                        status2xx += 1
                    elif status == 4:
                        status4xx += 1
                    elif status == 5:
                        status5xx += 1
            elif current_log:
                if 'text' not in current_log['response']:
                    current_log['response']['text'] = ''
                current_log['response']['text'] += line

                if "</html>" in line:
                    log_data.append(current_log)
                    current_log = {}
                elif "Error" in line:
                    log_data.append(current_log)
                    current_log = {}
                elif 'has_read' in current_log['response']:
                    log_data.append(current_log)
                    current_log = {}

    result[0] = result[0] + str(status2xx + status4xx + status5xx) + ',' + str(status2xx) + ',' + str(
        status4xx) + ',' + str(status5xx) + ','
    print("Total: " + str(status2xx + status4xx + status5xx))
    print("Status 2xx: " + str(status2xx))
    print("Status 4xx: " + str(status4xx))
    print("Status 5xx: " + str(status5xx))


    return log_data


def count_unique_500_errors(log_data,result_folder):
    unique_stack_traces = Counter()
    error_index = 0

    os.makedirs(result_folder, exist_ok=True)
    # Define the file path
    # file_path = os.path.join(result_folder, '/bug_to_request.csv')

    with open(
            result_folder+'/bug_to_request.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Request", "MuCoRest"])
        for log_item in log_data:
            if 'response' in log_item and "status_code" in log_item['response']:
                status_code = log_item['response']['status_code']
                response_text = log_item['response']['text']
                request_index = log_item['index']
                if status_code // 100 == 5:

                    if "stackTrace" in response_text:
                        response_text = response_text[response_text.find('"stackTrace"'):]
                        response_text = response_text[:response_text.find('java.lang.Thread')]
                        response_text = response_text[:response_text.find('Thread.java')]
                    elif "<title>" in response_text:
                        response_text = response_text[response_text.find("<title>"):response_text.find("</title>")]
                    elif "java:" in response_text:
                        response_text = re.findall(r"\w+\.java:\d+", response_text)
                        response_text = ', '.join(response_text)
                    else:
                        response_text = response_text[response_text.find("Error:"):]
                        response_text = re.sub(r'\[.*?\]', '', response_text)  # Remove words in square brackets
                        response_text = re.sub(r'\(.*?\)', '', response_text)  # Remove words in round brackets
                        response_text = re.sub(r'\'(.*?)\'|"(\1)"', '',
                                               response_text)  # Remove words in single or double quotes

                    error_message = response_text.strip()
                    # # dựa vào  error_message

                    if error_message not in unique_stack_traces:
                        error_index += 1

                    unique_stack_traces[error_message] += 1

                    full_stack_traces[error_message] = log_item['response']['text']
                csv_writer.writerow([str(request_index), str(error_index)])

    return unique_stack_traces


def get_missing_items(counter_a, counter_b):
    """
  This function takes two Counter objects and returns a list of items that are present in counter_a but not in counter_b.

  Args:
    counter_a: The first Counter object.
    counter_b: The second Counter object.

  Returns:
    A list of items that are present in counter_a but not in counter_b.
  """
    missing_items = []
    for item, count in counter_a.items():
        if item not in counter_b or counter_b[item] == 0:
            missing_items.append(item)
    return missing_items


# Example usage

def convert_err_stacktrace_to_dict(stacktrace):
    dict = {}
    stacktrace_components = stacktrace.split("\n")
    dict['method'] = stacktrace_components[0]
    dict['url'] = stacktrace_components[1]
    dict['body'] = stacktrace_components[2]
    dict['stacktrace'] = json.loads(stacktrace_components[3])
    return dict


if __name__ == '__main__':
    result_folder = sys.argv[1]
    service = sys.argv[2]
    print(result_folder)
    logs = ["features.txt", "languagetool.txt", "ncs.txt", "restcountries.txt", "scs.txt", "genome.txt", "person.txt",
            "user.txt", "market.txt", "project.txt"]
    csvs = ["_11000_1.csv", "_11010_1.csv", "_11020_1.csv", "_11030_1.csv", "_11040_1.csv", "_11050_1.csv",
            "_11060_1.csv", "_11070_1.csv", "_11080_1.csv", "_11090_1.csv"]
    result = [""]
    full_stack_traces = {}
    errors = {}
    code_coverage_csv_file = "code_coverage_" + service + ".csv"
    log_file = service + ".txt"
    errors[log_file] = []

    log_data = parse_log_file(log_file)
    unique_stack_traces = count_unique_500_errors(log_data, result_folder)
    unique_500_count = 0
    for stack_trace, count in unique_stack_traces.items():
        errors[log_file].append(full_stack_traces[stack_trace])
        unique_500_count += 1
    print(f'\nTotal unique number of 500 errors: {unique_500_count}')
    result[0] = result[0] + str(unique_500_count) + '\n'
    subprocess.run("mv " + log_file + " " + result_folder + "/" + log_file, shell=True)

    total_branch = 0
    covered_branch = 0
    total_line = 0
    covered_line = 0
    total_method = 0
    covered_method = 0
    with open(code_coverage_csv_file) as f:
        lines = f.readlines()
        for line in lines:
            items = line.split(",")
            if '_COVERED' not in items[6] and '_MISSED' not in items[6]:
                covered_branch = covered_branch + int(items[6])
                total_branch = total_branch + int(items[6]) + int(items[5])
                covered_line = covered_line + int(items[8])
                total_line = total_line + int(items[8]) + int(items[7])
                covered_method = covered_method + int(items[12])
                total_method = total_method + int(items[12]) + int(items[11])

    print("Code coverage: ", covered_branch / total_branch * 100, covered_line / total_line * 100,
          covered_method / total_method * 100)
    result[0] = result[0] + str(covered_method / total_method * 100) + ',' + str(
        covered_branch / total_branch * 100) + ',' + str(covered_line / total_line * 100) + '\n'
    subprocess.run("mv " + code_coverage_csv_file + " " + result_folder + "/" + code_coverage_csv_file, shell=True)
    # with open("res.csv", "w") as f:
    #     f.write(result[0])
    #
    # json_errors = {}
    # for log_file in errors:
    #     if log_file == 'person.txt':
    #         log_file_errors = errors[log_file]
    #         json_errors[log_file] = []
    #         for error in log_file_errors:
    #             json_errors[log_file].append(convert_err_stacktrace_to_dict(error))
    # with open('errors.json', 'w') as f:
    #     json.dump(json_errors, f)


    plt.figure(figsize=(8, 6))
    try:
        df = pd.read_csv(result_folder+"/bug_to_request.csv")
    except FileNotFoundError:
        print(f"Error: File not found. Please check the filename and try again.")
        exit()

    request = list(range(0, 20000))  # Example X data
    number_of_MuCoRest = (df['MuCoRest'].dropna().to_list()[:20000])

    # print(number_of_MuCoRest)
    plt.plot(list(range(0, len(number_of_MuCoRest))), number_of_MuCoRest, label='MuCoRest')

    try:
        df = pd.read_csv("ARAT-RL.csv")
    except FileNotFoundError:
        print(f"Error: File not found. Please check the filename and try again.")
        exit()
    number_of_ARAT_RL = (df[service].dropna().to_list()[:20000])
    plt.plot(request, number_of_ARAT_RL, label='ARAT-RL')

    # Set the Y axis to increments of 10, 20, 30, etc.
    plt.yticks(np.arange(0, 130, 10))
    # plt.xticks(np.arange(0, 20000, 4000))
    # Label the axes
    plt.xlabel('Number of Requests')
    plt.ylabel('Number of Bugs Found in ' + service)

    # Show the legend
    plt.legend()

    # Show the grid
    # plt.grid(True)
    plt.savefig(os.path.join(result_folder, "graph.png"), format='png')
    # Display the plot
    plt.show()


