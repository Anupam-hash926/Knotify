from datetime import UTC, datetime

import pytest

from knotify.contracts import Issue


@pytest.fixture
def issue_factory():
    def make(number=2, day=2, repository="example/project", **changes):
        values = {
            "issue_id": str(number),
            "repository": repository,
            "number": number,
            "title": "Crash during login",
            "body": "A traceback appears after signing in.",
            "created_at": datetime(2026, 1, day, tzinfo=UTC),
            "url": f"https://github.com/{repository}/issues/{number}",
        }
        return Issue(**(values | changes))

    return make
