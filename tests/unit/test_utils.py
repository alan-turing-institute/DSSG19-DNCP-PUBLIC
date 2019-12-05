import pytest

from src.utils.utils import connect_to_database, open_yaml

def test_connect_to_database():

    import sqlalchemy

    assert isinstance(connect_to_database(), sqlalchemy.engine.base.Engine)
