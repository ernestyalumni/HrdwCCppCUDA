import pathlib
import sys


def add_Voltron_system_path():

    sys.path.insert(
        0,
        # Voltron/Scripts/Problems -> Voltron/
        str(pathlib.Path(__file__).resolve().parent.parent.parent))