# from pathlib import Path
# import os
# import sys
# source_path = Path(os.path.abspath(__file__)).parent.parent
# print(source_path)
# if source_path not in sys.path:
#     sys.path.insert(0, str(source_path))

import pytest

from src.pipeline.pipeline import run_experiment

def test_run_experiment():
    from pathlib import Path
    import os
    import sys
    source_path = Path(os.path.abspath(__file__)).parent.parent.parent

    experiment = source_path / 'experiments' / 'dummy_experiment.yaml'

    try:
        run_experiment(experiment)
    except Exception as e:
        pytest.fail(e)
