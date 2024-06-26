

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
Change to Java 8: `. ./java8.env` to test Person Controller and Languagetool.
Make sure services you need to testing being in `services.json`.

I suggest you testing Person Controller and Languagetool separately because two different response time of two services lead to differrent numbers of requests. 





After installing all the required software, you make sure that `_11060_1.csv`, `_11010_1.csv` and `jacoco_11060_1.exec` doesnt exist and you can run the tools with this command to testing `person-controller`:

```
python run.py mucorest
```



### Testing services:
After run above command, use this command to enter the testing session:

#### Person Controller:
```
tmux a -t mucorest_person-controller
```
#### Language Tool:

```
tmux a -t languagetool
```



Make sure that having 4 session running, using this command to check:
```
tmux ls
```

Run this command to prevent duplicate sessions in future runs:
```
tmux kill-server
```

## Review the results

At the end of testing process, result will be saved at `experiment/{service_name}/{testing_start_time}`

Contains 5 files:
+ code_coverage_{service_name}.csv
+ bug_to_request.csv
+ graph.png
+ parameters.json
+ person.txt


