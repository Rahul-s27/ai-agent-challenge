import pandas as pd
import camelot
import numpy as np

def parse(pdf_path: str) -> pd.DataFrame:
    """
    Parses an ICICI bank statement PDF and returns a pandas.DataFrame
    with normalized transactions.

    Args:
        pdf_path (str): The path to the PDF bank statement.

    Returns:
        pandas.DataFrame: A DataFrame containing the normalized transactions
                          with columns: 'Date', 'Description', 'Debit Amt',
                          'Credit Amt', 'Balance'.
    """
    TARGET_COLUMNS = ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']

    COLUMN_MAPPING = {
        'date': 'Date',
        'txn date': 'Date',
        'description': 'Description',
        'debit': 'Debit Amt',
        'debit amt': 'Debit Amt',
        'credit': 'Credit Amt',
        'credit amt': 'Credit Amt',
        'balance': 'Balance',
    }

    all_extracted_dfs = []

    try:
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream', row_tol=10, edge_tol=100)
        if not tables:
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
        if not tables:
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream', edge_tol=250)
        if not tables:
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream', row_tol=20)
        if not tables:
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream', split_text=True)

    except Exception:
        return pd.DataFrame(columns=TARGET_COLUMNS)

    for table in tables:
        df = table.df
        header_row_idx = -1
        for i, row in df.iterrows():
            row_text = " ".join(row.astype(str).fillna('').str.lower().str.strip().tolist())
            has_date = any(k in row_text for k in ['date', 'txn date', 'transaction date'])
            has_desc = any(k in row_text for k in ['description', 'particulars', 'narration'])
            has_amount = any(k in row_text for k in ['debit', 'credit', 'balance', 'amt', 'amount'])

            if has_date and has_desc and has_amount:
                header_row_idx = i
                break

        if header_row_idx == -1:
            continue

        header = df.iloc[header_row_idx].astype(str).str.lower().str.strip()
        df = df[header_row_idx + 1:].copy()
        df.columns = header

        new_columns = []
        for col in df.columns:
            mapped_col = COLUMN_MAPPING.get(col, col)
            new_columns.append(mapped_col)
        df.columns = new_columns

        current_cols = [col for col in TARGET_COLUMNS if col in df.columns]
        if not current_cols:
            continue
        
        df = df[current_cols]
        
        numeric_cols_in_df = [col for col in ['Debit Amt', 'Credit Amt', 'Balance'] if col in df.columns]
        if numeric_cols_in_df:
            for col in numeric_cols_in_df:
                df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)
            df = df.dropna(how='all', subset=numeric_cols_in_df)
        
        df = df[df.apply(lambda x: x.astype(str).str.strip().ne('').sum() > 1, axis=1)]

        if not df.empty:
            all_extracted_dfs.append(df)

    if not all_extracted_dfs:
        return pd.DataFrame(columns=TARGET_COLUMNS)

    final_df = pd.concat(all_extracted_dfs, ignore_index=True)

    if 'Date' in final_df.columns:
        final_df['Date'] = final_df['Date'].astype(str).str.strip()
        date_formats = ['%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d']
        parsed_dates = pd.Series(pd.NaT, index=final_df.index)
        for fmt in date_formats:
            unparsed_mask = parsed_dates.isna()
            if unparsed_mask.any():
                attempt_parse = pd.to_datetime(final_df.loc[unparsed_mask, 'Date'], format=fmt, errors='coerce')
                parsed_dates.loc[unparsed_mask] = parsed_dates.loc[unparsed_mask].fillna(attempt_parse)
        final_df['Date'] = parsed_dates
        final_df = final_df.dropna(subset=['Date'])
    else:
        return pd.DataFrame(columns=TARGET_COLUMNS)

    numeric_cols = ['Debit Amt', 'Credit Amt', 'Balance']
    for col in numeric_cols:
        if col in final_df.columns:
            final_df[col] = final_df[col].astype(str).str.replace(',', '', regex=False).str.strip()
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
        else:
            final_df[col] = np.nan

    for col in TARGET_COLUMNS:
        if col not in final_df.columns:
            if col == 'Description':
                final_df[col] = ''
            else:
                final_df[col] = np.nan

    final_df = final_df[TARGET_COLUMNS]

    final_df['Date'] = final_df['Date'].dt.strftime('%d-%m-%Y')

    return final_df