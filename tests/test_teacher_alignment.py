"""Review P0-1 — lock the privileged-teacher prompt-logprob alignment.

Pure-python re-enactment of the three verl stages, kept in exact sync with:
  1. extract_prompt_logprobs  (verl/workers/rollout/vllm_rollout/utils.py):
     drops vLLM's None row 0, appends one dummy row → returned row i is the
     model's distribution AT position i (= prediction for token i+1).
  2. the trace-opd privileged re-alignment (verl agent_loop.py patch):
     slice teacher rows [T-1 : -1], lay out (P-1) zeros | R rows | 1 dummy.
  3. no_padding_2_padding loss slicing (verl/workers/utils/padding.py):
     reads rows [P-1 : P+R-1] as the predictions for the R response tokens.

The invariant: for every response token j, the row the loss reads must be
the teacher's distribution computed at teacher position T+j-1 (the position
whose next-token prediction IS response token j).
"""


def vllm_rows(seq_len: int):
    """Raw vLLM prompt_logprobs semantics: row i = logprob of token i given
    the prefix; row 0 is None. Encode each row by the position it PREDICTS."""
    return [None] + [f"pred_tok_{i}" for i in range(1, seq_len)]


def verl_extract(seq_len: int):
    """Stage 1: skip row 0, append a dummy → row i predicts token i+1."""
    rows = vllm_rows(seq_len)[1:]
    rows.append("DUMMY")
    assert len(rows) == seq_len
    return rows


def realign(teacher_rows, P: int, T: int, R: int):
    """Stage 2: the fixed trace-opd layout."""
    response_rows = teacher_rows[T - 1 : -1]
    assert len(response_rows) == R
    return ["ZERO"] * (P - 1) + response_rows + ["DUMMY"]


def loss_slice(rows, P: int, R: int):
    """Stage 3: what the distillation loss actually reads."""
    return rows[P - 1 : P + R - 1]


def test_every_response_token_gets_its_teacher_row():
    for P, T, R in [(5, 9, 4), (3, 3, 1), (2, 30, 7), (10, 11, 2)]:
        teacher_rows = verl_extract(T + R)
        laid_out = realign(teacher_rows, P=P, T=T, R=R)
        assert len(laid_out) == P + R
        window = loss_slice(laid_out, P=P, R=R)
        # response token j sits at sequence position T+j; its prediction is
        # made at position T+j-1, whose extracted row is named pred_tok_{T+j}.
        expected = [f"pred_tok_{T + j}" for j in range(R)]
        assert window == expected, (P, T, R, window)
        # nothing in the loss window may be a zero or dummy row
        assert "ZERO" not in window and "DUMMY" not in window


def test_old_slicing_fed_zero_to_first_response_token():
    """Document the pre-fix defect: slicing from T (instead of T-1) put a
    zero row where the FIRST response token's teacher distribution belongs."""
    P, T, R = 5, 9, 4
    teacher_rows = verl_extract(T + R)
    old_layout = ["ZERO"] * P + teacher_rows[T:]
    window = loss_slice(old_layout, P=P, R=R)
    assert window[0] == "ZERO"                      # the bug
    assert window[1:] == [f"pred_tok_{T + j}" for j in range(1, R)]


def test_non_privileged_passthrough_is_already_aligned():
    """teacher_prompt_ids=None (T==P) keeps the raw extraction — verify the
    official layout is aligned without any re-arrangement."""
    P, R = 6, 3
    rows = verl_extract(P + R)
    window = loss_slice(rows, P=P, R=R)
    assert window == [f"pred_tok_{P + j}" for j in range(R)]
