# CARLA Autonomous Driving

Simple implementation of autonomous driving on CARLA (part of CSE144 project)

## Prerequisites

- Python 3.12 (lower version may work but not tested)
- CARLA (TODO: add version)

## Installation

1. Clone the repository
2. Create a virtual environment (so that Python3.12 won't complain about breaking system dependencies)
```sh
python3.12 -m venv .carla
```
3. Activate the virtual environment
```sh
source .carla/bin/activate
```
4. Install the required packages
```sh
pip install -r requirements.txt
```

## Inference

```sh
python inference.py
```

