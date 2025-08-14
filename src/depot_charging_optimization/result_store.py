import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path


class ResultStore:
    def __init__(self, file: Path):
        self.file: Path = file

    def write(self, data: dict):
        msg_dict = OrderedDict()
        msg_dict["time"] = datetime.now().isoformat()
        msg_dict.update(data)
        self.file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file, "a") as f:
            f.write(json.dumps(msg_dict) + "\n")
