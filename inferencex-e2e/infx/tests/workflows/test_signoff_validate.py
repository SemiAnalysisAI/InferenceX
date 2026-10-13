"""Exercise the verifier's local check of its verdict file before publication."""

from infx.workflows import signoff_validate

WARN = "## ⚠️ **Verdict: WARN** ⚠️"


def write_verdict(tmp_path, rows):
    path = tmp_path / "verdict.md"
    path.write_text("\n\n".join(rows) + "\n", encoding="utf-8")
    return path


def passing_rows():
    return [f"✅ Check {number} (Requirement): PASS — Verified." for number in range(17)]


def test_publishable_verdict_reports_valid(tmp_path, capsys):
    rows = passing_rows()
    # The publisher rewrites ✅ to ➖ on N/A rows, so the check accepts it too.
    rows[12] = "✅ Check 12 (Append-only): N/A — No append-only entry."
    rows[16] = "⚠️ Check 16 (NVIDIA overview): WARN — Missing listing."
    path = write_verdict(tmp_path, [WARN, *rows])

    assert signoff_validate.main([str(path)]) == 0
    assert capsys.readouterr().out == "Verdict format is valid.\n"


def test_each_formatting_slip_is_reported_with_its_line(tmp_path, capsys):
    rows = passing_rows()
    rows[12] = "❌ Check 12 (Append-only): N/A — No append-only entry."
    rows[13] = "✅ Check 13 (Draft as shipped): PASS\n- **Draft:** the bundled head."
    rows[16] = "⚠️ Check 16 (NVIDIA overview): WARN — Missing listing."
    del rows[4]
    # Line 1 is the header; each row and the blank line after it take two lines.
    path = write_verdict(tmp_path, ["## ✅✅✅ **Verdict: PASS** ✅✅✅", *rows])

    assert signoff_validate.main([str(path)]) == 1
    assert capsys.readouterr().out.splitlines() == [
        "Line 25: N/A rows start with ➖.",  # noqa: RUF001 - verifier's N/A emoji
        "Line 27: write each check on one line as "
        "`<emoji> Check N (<name>): <STATUS> — <reason>`: ✅ Check 13 (Draft as shipped): PASS",
        "Missing Check 4, 13; include each of Checks 0-16 once.",
        f"The first line must be the only verdict header: {WARN}",
    ]


def test_unreadable_verdict_is_a_usage_error(tmp_path, capsys):
    assert signoff_validate.main([str(tmp_path / "missing.md")]) == 2
    assert capsys.readouterr().out == ""
