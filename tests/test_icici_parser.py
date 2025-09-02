import pandas as pd

from custom_parsers import icici_parser


def test_icici_parser_exact_match():
    def exists(p):
        try:
            open(p, 'rb').close()
            return True
        except FileNotFoundError:
            return False

    pdf_candidates = [
        "data/icici/icici_sample.pdf",
        "data/icici/icici sample.pdf",
    ]
    csv_candidates = [
        "data/icici/icici_sample.csv",
        "data/icici/result.csv",
    ]

    pdf_path = next((p for p in pdf_candidates if exists(p)), None)
    csv_path = next((p for p in csv_candidates if exists(p)), None)

    assert pdf_path is not None, "Sample PDF not found"
    assert csv_path is not None, "Sample CSV not found"

    expected = pd.read_csv(csv_path)
    got = icici_parser.parse(pdf_path)

    assert list(got.columns) == list(expected.columns)
    assert expected.equals(got)
