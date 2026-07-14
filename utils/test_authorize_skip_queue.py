from authorize_skip_queue import active_label_actor, is_authorized


class FakeApi:
    def __init__(self, events, memberships):
        self.events = events
        self.memberships = memberships
        self.membership_requests = []

    def paged(self, path):
        assert path == "/repos/SemiAnalysisAI/InferenceX/issues/42/timeline"
        return self.events

    def request(self, path):
        actor = path.rsplit("/", 1)[-1]
        self.membership_requests.append(actor)
        return self.memberships.get(actor, {"state": "inactive"})


def labeled(actor):
    return {
        "event": "labeled",
        "label": {"name": "skip_queue"},
        "actor": {"login": actor},
    }


def unlabeled(actor):
    return {
        "event": "unlabeled",
        "label": {"name": "skip_queue"},
        "actor": {"login": actor},
    }


def authorize(api):
    return is_authorized(
        api,
        repository="SemiAnalysisAI/InferenceX",
        pr_number=42,
        organization="SemiAnalysisAI",
        team_slug="core",
        label_name="skip_queue",
    )


def test_active_label_from_active_core_member_is_authorized():
    api = FakeApi([labeled("alice")], {"alice": {"state": "active"}})

    assert authorize(api) == (True, "alice")


def test_active_label_from_nonmember_is_rejected():
    api = FakeApi([labeled("mallory")], {"mallory": {"state": "inactive"}})

    assert authorize(api) == (False, "mallory")


def test_latest_labeling_actor_controls_after_remove_and_readd():
    events = [labeled("alice"), unlabeled("alice"), labeled("mallory")]
    api = FakeApi(
        events,
        {
            "alice": {"state": "active"},
            "mallory": {"state": "inactive"},
        },
    )

    assert active_label_actor(events, "skip_queue") == "mallory"
    assert authorize(api) == (False, "mallory")
    assert api.membership_requests == ["mallory"]


def test_removed_label_is_not_authorized_or_looked_up():
    api = FakeApi([labeled("alice"), unlabeled("bob")], {"alice": {"state": "active"}})

    assert authorize(api) == (False, None)
    assert api.membership_requests == []
