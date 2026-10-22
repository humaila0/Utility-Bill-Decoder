# WTQ Utility Bill Decoder

A Python-based solution for the **Women Tech Quest 2026 Build Track** challenge. This project extracts structured information from utility bill images and answers questions about the bills using Gemini and local OCR support.

## Features

- Processes the supplied utility bill images.
- Uses Google's Gemini multimodal model to analyze bill images.
- Generates Level 1 structured bill data and Level 2 question answers.
- Uses RapidOCR to provide additional text evidence to the model.
- Writes the required `level1.csv` and `level2.csv` output files.
- Preserves the original CSV templates, row order, and question text.
- Includes a verification script to help review generated outputs.

## Technology Stack

- **Python:** 3.11+
- **Google Gen AI SDK:** `google-genai`
- **AI model:** The Gemini model configured in `main.py`
- **OCR:** RapidOCR for local text extraction
- **Image processing:** Pillow
- **Environment configuration:** `python-dotenv`
- **CSV and JSON processing:** Python standard library

## Project Structure

```text
WTQ_Utility_Bill_Decoder/
├── data/
│   └── test/
│       ├── bills/
│       └── csv/
│           ├── level1.csv
│           └── level2.csv
├── output/
│   ├── level1.csv
│   └── level2.csv
├── source/
│   ├── main.py
│   ├── ocr_helper.py
│   ├── verify_output.py
│   ├── requirements.txt
│   ├── .env.example
│   ├── .gitignore
│   └── README.md
├── .gitignore
└── README.md
```

The `data/test/` folder contains the input bill images and original CSV templates. The `output/` folder contains the generated results.

## Prerequisites

- Python 3.11 or later
- A Google Gemini API key
- Internet access for Gemini API requests

## Setup on Windows

Open PowerShell in the project root directory.

Check your Python installation:

```powershell
py -3.11 --version
```

Create and activate the virtual environment if you have not already created it:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
```

Install the project dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r source/requirements.txt
```

If PowerShell blocks environment activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv311\Scripts\Activate.ps1
```

## Prepare the Competition Data

Download the competition data from the official participant guide:

https://tinyurl.com/wtq26

Place the test bill images and CSV templates in the following locations:

```text
data/test/bills/
data/test/csv/level1.csv
data/test/csv/level2.csv
```

Keep the original CSV templates unchanged. Ensure that each bill image filename matches its corresponding `bill_id`.

## Configure the API Key

1. Open Google AI Studio: https://aistudio.google.com/
2. Create a Gemini API key.
3. Copy `source/.env.example` to `source/.env`.
4. Add your API key to the `API_KEY` variable in `source/.env`.

Example configuration:

```text
API_KEY=your_gemini_api_key_here
```

Never commit your real `.env` file or expose your API key in source code, screenshots, or public repositories.

## Run the Project

From the project root, activate the virtual environment and run:

```powershell
python source/main.py --data-dir data --output-dir output
```

The program processes the input bills and generates the two required CSV files. Requests may be delayed and retried to reduce the impact of temporary API errors or rate limits.

## Verify the Output

Run the verification script to inspect the generated results:

```powershell
python source/verify_output.py --input-dir output --output-dir output_verified
```

Review the command-line options supported by your current version of the verification script before running it.

Expected output files:

- `output/level1.csv` — one row per bill, with columns `bill_id,json`.
- `output/level2.csv` — bill questions and answers, with columns `bill_id,question_id,question,answer`.

Always compare the generated values against the original bill images. OCR and AI extraction may produce mistakes, especially for tilted, low-resolution, or Urdu-heavy bills. Do not replace unreadable values with guesses.

## Security and GitHub

The root `.gitignore` should exclude environment files, virtual environments, and Python cache files.

Do not commit:

- `.env` files containing API keys
- `.venv311/`
- `__pycache__/` and `.pyc` files
- Temporary output files that are not intended for publication

Keep `.env.example` in the repository, without any real credentials.

## AI Usage Disclosure

- **AI model:** The Gemini model configured in `source/main.py`.
- **API:** Google Gemini API through the Google Gen AI Python SDK.
- **Local OCR:** RapidOCR.
- **Image processing:** Pillow.
- **AI coding assistance:** ChatGPT was used to help draft and organize parts of the code.
- **Other LLMs or hosted OCR services:** Declare any additional services actually used in the final implementation.

Update this disclosure if the implementation changes.
