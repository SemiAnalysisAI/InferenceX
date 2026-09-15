import tempfile
from pathlib import Path

import pytest
import yaml
from hypothesis import given, strategies as st

from infx.matrix.validation import load_config_files


@given(value=st.one_of(st.none(), st.booleans(), st.integers(), st.text(), st.lists(st.integers())))
def test_invalid_config_root_has_a_usable_validation_error(value):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "master.yaml"
        path.write_text(yaml.safe_dump(value))
        with pytest.raises(ValueError, match="must contain a dictionary"):
            load_config_files([str(path)], validate=False)


@given(key=st.one_of(st.none(), st.booleans(), st.integers()))
def test_non_string_config_keys_have_a_usable_validation_error(key):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "master.yaml"
        path.write_text(yaml.safe_dump({key: {}}))
        with pytest.raises(ValueError, match="key.*string"):
            load_config_files([str(path)], validate=False)
