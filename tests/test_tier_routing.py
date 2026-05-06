from datetime import date

import pytest

from scripts.config import get_tier_for_date


class TestRollingWindowTiers:
    def test_today_is_hot(self):
        today = date.today()
        assert get_tier_for_date(today, as_of=today) == "hot"

    def test_recent_event_is_hot(self):
        as_of = date(2026, 5, 1)
        recent = date(2025, 6, 1)  # ~11 months ago
        assert get_tier_for_date(recent, as_of=as_of) == "hot"

    def test_18mo_boundary_is_hot(self):
        as_of = date(2026, 5, 1)
        # 17 months ago → hot
        event = date(2024, 12, 1)
        assert get_tier_for_date(event, as_of=as_of) == "hot"

    def test_just_past_18mo_is_warm(self):
        as_of = date(2026, 5, 1)
        # ~24 months ago → warm
        event = date(2024, 5, 1)
        assert get_tier_for_date(event, as_of=as_of) == "warm"

    def test_within_78mo_is_warm(self):
        as_of = date(2026, 5, 1)
        # ~60 months ago → warm
        event = date(2021, 5, 1)
        assert get_tier_for_date(event, as_of=as_of) == "warm"

    def test_older_than_78mo_is_cold(self):
        as_of = date(2026, 5, 1)
        # ~96 months ago → cold
        event = date(2018, 5, 1)
        assert get_tier_for_date(event, as_of=as_of) == "cold"

    def test_string_input_accepted(self):
        assert get_tier_for_date("2026-04-01", as_of="2026-05-01") == "hot"

    def test_default_as_of_is_today(self):
        # Just check it doesn't crash and returns a valid tier
        result = get_tier_for_date(date.today())
        assert result in ("hot", "warm", "cold")

    def test_future_event_raises(self):
        with pytest.raises(ValueError, match="after as_of"):
            get_tier_for_date(date(2030, 1, 1), as_of=date(2026, 5, 1))

    def test_rolling_window_shifts(self):
        # Same event, different as_of → different tier
        event = date(2024, 5, 1)
        # As of May 2026: 24 months ago → warm
        assert get_tier_for_date(event, as_of=date(2026, 5, 1)) == "warm"
        # As of May 2025: 12 months ago → hot
        assert get_tier_for_date(event, as_of=date(2025, 5, 1)) == "hot"
