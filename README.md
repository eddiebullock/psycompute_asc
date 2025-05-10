# Autism Screening Assessment Pipeline

A data processing and analysis pipeline for autism screening assessments (AQ, SQ, EQ).

## Setup

1. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the data processing pipeline:
```bash
python src/data/pipeline.py
```

2. View the interactive dashboard:
```bash
streamlit run src/visualization/dashboard.py
```

## Project Structure

```
bulldev/
├── data/
│   ├── raw/           # Original data files
│   └── processed/     # Processed data and visualizations
├── src/
│   ├── data/
│   │   ├── generation/    # Data generation code
│   │   ├── processing/    # Data processing code
│   │   ├── validation/    # Data validation code
│   │   ├── analysis/      # Analysis and visualization code
│   │   └── pipeline.py    # Main pipeline script
│   └── visualization/     # Dashboard code
├── tests/               # Unit tests
├── requirements.txt     # Project dependencies
└── README.md           # This file
```

## Development

- All code should be in the `src` directory
- Tests should be in the `tests` directory
- Data files should be in the `data` directory
- Use the virtual environment for all development work

## License

to come...