# Scripting
- [Scripting](#scripting)
  - [Command Line](#command-line)
    - [Useful Commands](#useful-commands)
  - [Organization](#organization)
  - [Packaging](#packaging)
    - [Virtual Environment](#virtual-environment)
    - [Setup File](#setup-file)
      - [requirements.txt](#requirementstxt)
      - [Additional Packages](#additional-packages)
      - [Entry Points](#entry-points)
  - [Documentation](#documentation)
    - [reighns_mnist\config.py](#reighns_mnistconfigpy)
  - [Logging](#logging)
    - [Intuition](#intuition)
    - [Levels](#levels)



## Command Line
1. Open up PowerShell.
2. `mkdir mnist`
3. `cd mnist`
4. `code .` to open VSCode.
5. `code . mlops.md` to create or open the file `mlops.md` if it exists.

### Useful Commands

  sudo apt-get upgrade
  sudo apt-get install
  sudo apt-get remove

  ls  # list all files and folders in current directory

  cd  # change directory
  cd / # root directory
  cd .. # up one directory level
  cd - # previous directory

  pwd # print working directory

  mkdir folder_name # creates folder at current directory
  mkdir folder_name/folder_name # creates folder in another folder in current directory

  history # previous commands

  df # display file system - disk space

  du # directory usage

  free # amount of free space

  crontab -l # shows list of cron jobs
  crontab -e # edit cron jobs select 1

  git clone https://github.com/edtyg/slack # clone new repository

  **to set up git**
  sudo apt-get install git

  **to set up pip**

  sudo apt-get install software-properties-common
  sudo apt-add-repository universe
  sudo apt-get update
  sudo apt-get install python-pip

  pip install pandas
  pip install slack
  pip install slackclient




## Organization


---

## Packaging

### Virtual Environment

1. `python -m venv mnist` to create a new virtual environment.
2. `mnist\Scripts\activate` on command prompt or powershell to activate the virtual environment. Type `deactivate` to exit our virtual environment.
3. `python -m pip install --upgrade pip setuptools wheel` to install the latest pip.

### Setup File

Execute in the following manner:

```
pip install -e . -f https://download.pytorch.org/whl/torch_stable.html  # installs required packages only      
python -m pip install -e ".[dev]"                                       # installs required + dev packages
python -m pip install -e ".[test]"                                      # installs required + test packages
python -m pip install -e ".[docs_packages]"                             # installs required documentation packages
```

#### requirements.txt

1. Usual users will just make use of this text file to download all the required packages.
2. Author does not recommend using `pip freeze` to write libraries to `requirements.txt`.
    > We've been adding packages to our requirements.txt as we've needed them but if you haven't, you shouldn't just do pip freeze > requirements.txt because it dumps the dependencies of all your packages into the file. When a certain package updates, the stale dependency will still be there. To mitigate this, there are tools such as pipreqs, pip-tools, pipchill, etc. that will only list the packages that are not dependencies. However, if you're separating packages for different environments, then these solutions are limited as well.
3. Something worth taking note is when you download PyTorch Library, there is a dependency link, you may execute as such: 
   ```
   pip install -e . -f https://download.pytorch.org/whl/torch_stable.html
   ```
---
#### Additional Packages

1. For developers, you may also need to use `test_packages`, `dev_packages` and `docs_packages` as well.


#### Entry Points

The final lines of the file define various entry points we can use to interact with the application. Here we define some console scripts (commands) we can type on our terminal to execute certain actions. For example, after we install our package, we can type the command `hn` to run the app variable inside mnist/cli.py.

```
    entry_points={
        "console_scripts": [
            "tagifai = app.cli:app",
        ],
    },
```

## Documentation

### reighns_mnist\config.py

Read the documentation in the file itself.


---

## Logging

### Intuition

Logging the process of tracking and recording key events that occur in our applications. We want to log events so we can use them to inspect processes, fix issues, etc. They're a whole lot more powerful than print statements because they allow us to send specific pieces of information to specific locations, not to mention custom formatting, shared interface with other Python packages, etc. We should use logging to provide insight into the internal processes of our application to notify our users of the important events that are occurring.

### Levels

Before we create our specialized, configured logger, let's look at what logged messages even look like by using a very basic configuration.

```
import logging
import sys

# Create super basic logger
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Logging levels (from lowest to highest priority)
logging.debug("Used for debugging your code.")
logging.info("Informative messages from your code.")
logging.warning("Everything works but there is something to be aware of.")
logging.error("There's been a mistake with the process.")
logging.critical(
    "There is something terribly wrong and process may terminate.")
```