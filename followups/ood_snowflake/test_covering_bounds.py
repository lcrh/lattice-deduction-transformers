"""Covering-grid bounds must stay aligned between data.py and gen_data.py."""

from experiments.snowflake import data as snowflake_data
from experiments.snowflake import gen_data as snowflake_gen_data


def test_covering_bounds_match_between_data_and_gen_data():
    assert snowflake_gen_data.COVER_Q_MIN == snowflake_data.Q_MIN
    assert snowflake_gen_data.COVER_Q_MAX == snowflake_data.Q_MAX
    assert snowflake_gen_data.COVER_R_MIN == snowflake_data.R_MIN
    assert snowflake_gen_data.COVER_R_MAX == snowflake_data.R_MAX
