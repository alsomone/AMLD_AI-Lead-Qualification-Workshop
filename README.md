# AMLD AI Lead Qualification Workshop
##From Marketing Leads to Explainable Sales Recommendations with AI
This hands-on workshop demonstrates how AI and LLM pipelines can help bridge the gap between Marketing and Sales by automatically evaluating incoming leads and identifying those with the highest business potential.

Starting from a lead submission form, the prototype combines structured customer information, AI-based analysis, and transparent business rules to generate an explainable lead qualification and recommend the next best action.

The objective is not to let the LLM make the commercial decision by itself. Instead, the workshop demonstrates how LLM reasoning and deterministic business logic can complement each other.

## 🎯 What You Will Build
During the workshop, we will explore a pipeline that transforms a marketing lead submission into an explainable commercial recommendation.

The solution will:

Capture lead information through a Streamlit form.
Analyze different aspects of the lead in parallel using an LLM.
Assess business fit, contact relevance, demand potential, purchasing intent, and missing information.
Calculate an auditable qualification score using deterministic business rules.
Generate an explanation of the qualification.
Recommend the next best action for Marketing or Sales.
Generate questions or a follow-up message when additional information is required.
## 🧠 Solution Concept
The prototype follows this general flow:

Marketing Lead
      │
      ▼
Streamlit Lead Form
      │
      ▼
LangChain Pipeline
      │
      ├───────────────┐
      ▼               ▼
Business Fit      Demand Analysis
Analysis
      │               │
      └───────┬───────┘
              ▼
     Deterministic Scoring
              │
              ▼
      Qualification Status
              │
              ▼
        Final LLM Synthesis
              │
              ▼
   Explanation + Next Action
The LLM analyzes qualitative information, while Python business rules calculate the qualification score.

This separation is intentional: AI provides interpretation, while deterministic rules provide transparency and auditability.

## 🛠️ Technology Stack
The workshop uses:

Python for the application and business logic
Streamlit for the interactive user interface
LangChain for LLM pipeline orchestration
Ollama for running the LLM locally
Mistral as the default local language model
GitHub for distributing and versioning the workshop project
The LLM runs locally through Ollama, so no paid LLM subscription or API credentials are required.

# 🚀 Before the Workshop
Participants who want to experiment with the code should prepare their environment before the workshop.

Participants who prefer not to install the environment can still follow the complete live demonstration on the presentation screen.

Prerequisites
Please install:

Python 3.11 or later
Visual Studio Code
Ollama
You should also download or clone this repository to your computer.

## 1. Clone the Repository
Open PowerShell and run:

git clone https://github.com/alsomone/AMLD_AI-Lead-Qualification-Workshop.git
Then enter the project folder:

cd AMLD_AI-Lead-Qualification-Workshop
## 2. Create a Python Virtual Environment
python -m venv .venv
This creates an isolated Python environment for the workshop.

## 3. Activate the Environment
On Windows PowerShell:

.\.venv\Scripts\Activate.ps1
When activated, your terminal should start with something similar to:

(.venv) PS C:\...\AMLD_AI-Lead-Qualification-Workshop>
## 4. Install the Python Dependencies
python -m pip install -r requirements.txt
This installs the packages required by the workshop, including Streamlit, LangChain, and the Ollama integration.

## 5. Download the Mistral Model
Make sure Ollama is installed and running.

Then execute:

ollama pull mistral
The model download is several GB, so please complete this step before the workshop.

## 6. Check Your Environment
The repository contains an automated environment check.

Run:

python setup_check.py
If everything is configured correctly, you should see:

AI Lead Qualification Workshop
======================================
[OK] Python
[OK] streamlit
[OK] langchain-core
[OK] langchain-ollama
[OK] Ollama is running
[OK] Mistral is available
======================================
[SUCCESS] The workshop environment is ready.
If you see:

[SUCCESS] The workshop environment is ready.
you are ready for the workshop.

## ▶️ Running the Application
Activate your virtual environment if it is not already active:

.\.venv\Scripts\Activate.ps1
Then start the Streamlit application:

streamlit run app.py
Streamlit will provide a local URL that you can open in your browser.

You can then submit a test lead and observe how the AI-assisted qualification pipeline evaluates it.

## 📁 Repository Structure
AMLD_AI-Lead-Qualification-Workshop/
│
├── app.py
│   Main Streamlit application and AI qualification pipeline
│
├── requirements.txt
│   Python dependencies required by the workshop
│
├── setup_check.py
│   Checks Python packages, Ollama, and Mistral
│
├── README.md
│   Workshop documentation and setup instructions
│
├── .gitignore
│   Prevents local environments and sensitive files from being committed
│
└── data/
    └── Lead.csv
        Synthetic lead data used for workshop examples
## 🔍 What to Explore During the Workshop
Participants will be encouraged to experiment with different parts of the solution.

For example:

### Change the lead information
Compare leads with different:

business types
contact roles
estimated demand
areas of interest
purchasing timeframes
levels of information completeness

### Explore the prompts
Observe how the LLM is instructed to:

use only information provided in the lead form
avoid inventing company information
identify missing information
provide evidence for its assessment

### Explore the scoring rules
The qualification score is calculated using deterministic Python rules rather than by the LLM.

This makes it possible to understand exactly why a lead receives a particular score.

### Compare AI reasoning with business rules
One of the key workshop questions is:

Which decisions should AI help interpret, and which decisions should remain controlled by explicit business rules?

## 🔒 Data and Privacy
This workshop is designed to run locally.

The prototype:

uses synthetic workshop data
runs the LLM locally through Ollama
does not require paid LLM API credentials
is designed for experimentation and learning
Do not use confidential, personal, customer, or commercially sensitive information when experimenting with the workshop.

## ⚠️ Prototype Disclaimer
This project is a workshop prototype.

The qualification criteria, scoring thresholds, prompts, and recommendations are illustrative examples and should not be considered production-ready commercial rules.

For a production implementation, qualification logic should be validated with the relevant Marketing, Sales, Data, Legal, Privacy, Security, and IT stakeholders.

## 🎓 Workshop Goal
By the end of the session, participants should understand how an AI-assisted pipeline can transform a marketing form submission into an explainable, auditable, and actionable commercial recommendation while combining the strengths of LLM reasoning and deterministic business logic.