from scripts.repro.prepare_egoschema_subset import canonicalize_egoschema_row


def test_canonicalize_egoschema_subset_row_maps_hf_fields():
    row = {
        "question_idx": "00000",
        "question": "What is C doing?",
        "video_idx": "0074f737-11cb-497d-8d07-77c3a8127391",
        "option": ["A. cooking", "B. laundry", "C. dishes", "D. cleaning", "E. shopping"],
        "answer": "3",
    }

    out = canonicalize_egoschema_row(row)

    assert out == {
        "question_idx": "00000",
        "question": "What is C doing?",
        "options": ["A. cooking", "B. laundry", "C. dishes", "D. cleaning", "E. shopping"],
        "correct_answer": "D",
        "video_path": "0074f737-11cb-497d-8d07-77c3a8127391.mp4",
    }


def test_canonicalize_egoschema_subset_row_accepts_letter_answer():
    row = {
        "question_idx": "q1",
        "question": "Which choice?",
        "video_idx": "vid1",
        "options": ["A. one", "B. two"],
        "answer": "B",
    }

    assert canonicalize_egoschema_row(row)["correct_answer"] == "B"
