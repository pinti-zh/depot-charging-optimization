# Depot Charging Optimization

![alt text](assets/title_image.png "Title")

## Installation

Follow these steps to install and set up the project locally.

### 1. Install Python (if necessary)

Make sure you have Python 3.13 or above installed on your system. You can check your Python version running the following command:

```bash
python3 --version
```

If you don't have Python 3.13 or above installed, you can download and install the latest version from the official Python website:

- [Download Python](https://www.python.org/downloads/)

### 2. Clone the Repository

Clone the repository to your local machine using the following command:

```bash
git clone git@github.com:pinti-zh/depot-charging-optimization.git
```

### Create a Python Virtual Environment

Navigate into the project directory:

```bash
cd depot-charging-optimization
```

Now, create a Python virtual environment. It's recommended to use `venv` for creating isolated environments:

**On macOS/Linux:**

```bash
python3 -m venv .venv
```

**On Windows:**

```bash
python -m venv .venv
```

This will create a new `.venv` directory in the project, containing the virtual environment.

### 4. Activate the Virtual Environment

Activate the virtual environment:

**On macOS/Linux:**

```bash
source .venv/bin/activate
```

**On Windows:**

```bash
.\venv\Scripts\activate
```

When activated, your shell prompt should change to show the virtual environment name, indicating that you're working inside the isolated environment.

### 5. Install the Project Dependencies

With the virtual environment activated, install the project dependencies using `pip`:

```bash
pip install .
```

With all the dependencies installed, the project is now ready to be used.

## Usage

### Unit Tests
Optionally, run the following command to run the unit tests.
```bash
pytest
```

### Cleaning the Data

To clean the data run
```bash
clean-data data/raw/vehicle_cycle_energy.csv data/clean/vehicle_cycle_energy.csv
```

### Preprocessing the Data
To preprocess the data run
```bash
preprocess-data data/clean/vehicle_cycle_energy.csv -t data/processed
```

### Optimization

Hyper-parameters can be adjusted in the config files inside the config/ directory. Alternatively, they can be supplied as command line arguments.
Run the following command to see which options there are.
```bash
optimize --help
```

Run the following command to run the optimization.
```bash
optimize
```

To see the results run

```bash
plot
```

The results will be shown as a dashboard in the browser.

