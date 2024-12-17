# Financial Analysis LLM Workflow & OFIN Benchmark

This repository contains the implementation of a systematic workflow for calculating financial matrices using Large Language Models (LLMs) and introduces OFIN, a benchmark for evaluating LLM capabilities in financial matrix calculation.

## Overview

The project presents a six-step end-to-end workflow designed to calculate financial matrices using generative AI and establishes a benchmark to evaluate various models' performance in this task.

### Key Findings
- Maximum overall accuracy rate: 31.25%
- Accuracy with ±10% tolerance: 71.25%
- Within the same model series, larger models consistently achieve higher accuracy
- Models perform better with single-step calculations and single-period data
- Performance decreases with tasks requiring average terms and multi-step calculations

## Workflow Steps

1. **Table Extraction**: Uses Camelot library to extract tables from financial reports
2. **Financial Statement Classification**: Identifies relevant financial statements using key subjects
3. **Time and Subject Labeling**: Processes column data to maintain time and subject context
4. **Formula Generation**: Generates formulas for financial matrices
5. **Value Extraction**: Extracts required numerical values
6. **Matrix Calculation**: Performs final calculations using extracted data

## Installation

```bash
pip install pandas camelot-py glob pathlib json logging
```

## Dependencies

- pandas
- camelot
- glob
- pathlib
- json
- logging
- aisuite (custom AI client)

## Usage

1. Set up your API keys:

```python
os.environ['OPENAI_API_KEY'] = "YOUR_API_KEY"
os.environ['ANTHROPIC_API_KEY'] = "YOUR_API_KEY"
os.environ['GOOGLE_API_KEY'] = "YOUR_API_KEY"
os.environ['MISTRAL_API_KEY'] = "YOUR_API_KEY"
```

2. Run the main pipeline:

```python
if __name__ == "__main__":
    pdf_folder = "YOUR_PDF_FOLDER"
    results_df = pd.DataFrame(columns=['Stkcd'] + matrix_list)
    
    formula_info = get_all_formulas(model, matrix_list)
    
    for stkcd, tables in tables_by_company.items():
        labeled_tables = add_labels_using_gpt(model, tables)
        all_values = extract_all_values(model, labeled_tables, formula_info)
        results = calculate_matrices(model, all_values, matrix_list)
        results_df = pd.concat([results_df, create_results_df(results, stkcd)])
```

## OFIN Benchmark

The OFIN benchmark consists of:
- 18 operating-related financial matrices
- Data from 5,442 Chinese listed companies
- Fiscal year 2023 data
- Sourced from CSMAR database

## Model Performance

Performance evaluation of different LLM models:

| Model | Exact Match | Within ±1% | Within ±5% | Within ±10% |
|-------|-------------|------------|------------|-------------|
| Claude-3.5 Sonnet | 31.25% | 50.00% | 57.50% | 71.25% |
| OpenAI O1 Preview | 27.78% | 38.89% | 44.44% | 55.56% |
| OpenAI O1 Mini | 20.00% | 20.00% | 26.25% | 41.25% |
| Claude-3.5 Haiku | 15.00% | 15.00% | 25.00% | 40.00% |
| OpenAI GPT-4o | 15.00% | 15.00% | 20.00% | 30.00% |
| OpenAI GPT-4o Mini | 6.25% | 6.25% | 10.00% | 16.25% |

## Data Availability

- Financial matrices data: [CSMAR Database](https://data.csmar.com/)
- Annual reports: Available in this repository

## Contributing

Please reference the project's code style and contribution guidelines before submitting pull requests.

## Acknowledgments

Special thanks to:
- Professor Thomas Bourveau
- Professor Tianyi Peng
- Professor Hannah Li
- Professor Stephen Penman and the MSAFA program at Columbia Business School

## Author

Haonan Chen (hc3441@columbia.edu)
