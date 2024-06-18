

# Set up the experiment


## Software Dependencies and Installation

If your OS is Ubuntu 20.04, you can simply run our setup script with `sh setup.sh` command in your terminal.

The following software is required for the experiment:
- Git
- Common utilities (software-properties-common, unzip, wget, gcc, git, vim, libcurl4-nss-dev, tmux, mitmproxy)
- Java 8 and 11
- Maven3
- Python 3 (with pip and virtualenv)
- Python libraries in requirements.txt
- Docker
- .NET 6 Runtime and SDK
- EvoMaster 1.6.0
- RESTler 9.1.1
- JaCoCo Agent and CLI 0.8.7

# Run the experiment

## Run tools and services
Change to Java 8: `. ./java8.env`

After installing all the required software, you make sure that `_11060_1.csv` and `jacoco_11060_1.exec` doesnt exist and you can run the tools with this command:

```
python run.py [tool's name]
```

This command will run the tool and all the services in our benchmark for an hour. Possible tool names are `mucorest`, `morest`, `evomaster-blackbox`, and `restler`.

### Run person-controller service
After run above command, use this command to enter the active session:
```
tmux attach-session -t arat-rl_person-controller
```
You can run this command to check if all 4 sessions are operating concurrently, use this command:
```
tmux ls
```

If the "arat-rl_person-controller" session ends abruptly, run this command to prevent duplicate sessions in future runs:
```
tmux kill-server
```

## Collect the results

To collect the results, use the following command:

```
python parse_log.py
```

This will gather the coverage and number of responses for status codes 2xx, 4xx, and 5xx. The results will be stored in the `res.csv` file. Additionally, any detected bugs will be recorded in the `errors.json` file.

## Review the Results

The `results` directory contains the results for each tool and each service. These results include the achieved code coverage, the number of obtained status codes, the number of bugs found, and detailed bug reports.