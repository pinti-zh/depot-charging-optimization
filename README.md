# Depot Charging Optimization

![alt text](assets/title_image.png "Title")

## Project Overview

This repository focuses on optimizing charging strategies for electric bus depots. The core objective is to determine when each bus should be charged and at what power level, ensuring operational readiness while minimizing energy costs.

For a comprehensive explanation of the methodology, assumptions, and results, please refer to the accompanying thesis:

👉 [Depot Charging Optimization for Electric Bus Fleets](https://doi.org/10.3929/ethz-c-000787335)

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
To use a battery add `data/battery.json` to data_files in `config/file.yaml`

### Plotting Results

To see the results run

```bash
dashboard
```

The results will be shown as a dashboard in the browser on localhost:8000.

### Model Predictive Control

Adjust the *confidence_level* and *energy_std_dev* in `config/optimizer.yaml` to 0.95 and 0.05. The optimizer will now perform robust optimzation.

To run model predictive control run
```bash
mpc
```

This will run MPC with uniform timesteps and re-optimize once per hour. The simulation will span one day.

## Acknowledgements

**Main Contributors:**
- Luca Pinter [IDSC]
- Fabio Widmer [IDSC]

### Special Thanks
We would also like to thank the following organizations and individuals for their support and contributions:

- **Organizations**
  - PostAuto Switzerland
  - Zurich Information Security & Privacy Center (ZISC)
  - Institute for Dynamic Systems and Control (IDSC), ETH Zürich

- **Individuals**
  - **Eric Imstepf [PostAuto]** - for making the collaboration with PostAuto seamless, always bringing enthusiasm and encouragement to the project.
  - **Julien Burri [PostAuto]** - for providing valuable feedback on real-world constraints during the project collaboration with PostAuto.
  - **Anina Leuch & Lars Schmutz [PostAuto]** - for providing data from PostAuto, offering valuable feedback, and consistently participating in our monthly meetings. 
  - **Dr. Kari Kostiainen [ZISC]** - for supporting the project through ZISC and making collaboration effortless by keeping formalities and bureaucracy to a minimum.
  - **Prof. Dr. Christopher Onder [IDSC]** – for his support as head of the IDSC research group.

## License

This project is released under the **MIT License**.
