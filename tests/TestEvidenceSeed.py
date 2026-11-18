# TestEvidenceSeed.py
# Evidence that does not fit one call is subsampled.  The draw is seeded and private: the same
# records give the same sample on every run, evidence_seed= picks a different one, and nothing
# reseeds or consumes the global random generator.
#
# Runs offline.  No API keys, no LLM calls, no network.
#
#     python tests/TestEvidenceSeed.py

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import theorizer as th


def many_records(count:int=120):
    records = []
    for index in range(count):
        records.append({"name_short": "finding " + str(index).zfill(3), "uuid": "e" + str(index) + ".0",
                        "brief_description": "A measured effect of moderate size, described at some length. " * 4})
    return records


# Small enough that roughly half the records have to go
BUDGET = 6000


def uuids(records):
    return [record["uuid"] for record in records]


def test_the_same_records_give_the_same_sample(directory):
    first, rate = th.Pipeline.subsample(many_records(), BUDGET)
    assert (rate < 1.0), "test premise: the records should not fit the budget"
    random.random()     # whatever else the program does with the global generator in between
    second, _ = th.Pipeline.subsample(many_records(), BUDGET)
    assert (uuids(first) == uuids(second)), "two runs over the same records drew different evidence"


def test_evidence_seed_picks_a_different_sample(directory):
    default, _ = th.Pipeline.subsample(many_records(), BUDGET)
    other, _ = th.Pipeline.subsample(many_records(), BUDGET, evidence_seed=12345)
    again, _ = th.Pipeline.subsample(many_records(), BUDGET, evidence_seed=12345)
    assert (uuids(other) == uuids(again)), "the same evidence_seed drew different evidence"
    assert (uuids(other) != uuids(default)), "evidence_seed did not change the sample"


def test_subsampling_leaves_the_global_generator_alone(directory):
    random.seed(2024)
    expected = [random.random() for _ in range(3)]

    random.seed(2024)
    th.Pipeline.subsample(many_records(), BUDGET)
    th.Pipeline.subsample(many_records(), BUDGET, evidence_seed=7)
    observed = [random.random() for _ in range(3)]
    assert (observed == expected), "subsampling used or reseeded the global random generator"


def test_the_type_sweep_does_not_reseed_the_global_generator(directory):
    def fake_build_theory_with_types(theory_query, records, **kwargs):
        return {"theory_response": {"theories_general": [], "theories_specific": []}, "total_cost": 0.0}

    th.Config.init(project_dir=None, verbose=False)
    real = th.Legacy.build_theory_with_types
    th.Legacy.build_theory_with_types = fake_build_theory_with_types
    try:
        random.seed(99)
        expected = [random.random() for _ in range(3)]
        random.seed(99)
        packed = th.Pipeline.generate_theories_over_types(many_records(), "a question", output_types=["ASSOCIATION"], inference_types=[],
                                                          max_data_tokens=BUDGET, evidence_seed=5, show_progress=False)
        observed = [random.random() for _ in range(3)]
    finally:
        th.Legacy.build_theory_with_types = real

    assert (observed == expected), "evidence_seed reseeded the global random generator"
    assert (packed["evidence_seed"] == 5), "the seed used was not recorded"


def main():
    print("TestEvidenceSeed.py")
    tests = [
        test_the_same_records_give_the_same_sample,
        test_evidence_seed_picks_a_different_sample,
        test_subsampling_leaves_the_global_generator_alone,
        test_the_type_sweep_does_not_reseed_the_global_generator,
    ]
    for test in tests:
        test(None)
        print("  " + test.__name__ + ": OK")

    print("TestEvidenceSeed.py: " + str(len(tests)) + " test(s) passed")
    return 0


if (__name__ == "__main__"):
    sys.exit(main())
