import json
from typing import Any, Dict, List

import streamlit as st

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

#from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="GePeTto AI Lead Qualification",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 GePeTto AI Lead Qualification")

st.markdown(
    """
This prototype demonstrates how an open-source LLM pipeline can support the
qualification of a lead submitted through a marketing form.

**Features:**
- ✅ Completely free and open-source
- 🖥️ Runs locally (no external API calls or internet required)
- 🔒 Your data stays on your computer
- ⚡ Powered by Ollama + Mistral/Llama models

The solution combines:

1. Form validation
2. Parallel AI analysis
3. Transparent business rules
4. Deterministic scoring
5. An AI-generated explanation and recommended next action
"""
)


# ============================================================
# 2. LOAD CREDENTIALS AND INITIALIZE THE MODEL
# ============================================================

# Using free, open-source Ollama models
# No credentials needed!

OLLAMA_MODELS = [
    "llama2",           # Meta's Llama 2 (good general purpose)
    "mistral",          # Mistral (fast and accurate)
    "neural-chat",      # Intel's Neural Chat (optimized)
    "dolphin-mixtral",  # Dolphin Mixtral (strong reasoning)
]

OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama server URL


@st.cache_resource
def initialize_llm(model_name: str = "mistral") -> ChatOllama:
    """
    Initialize and cache a free Ollama LLM model.
    
    First, ensure Ollama is running:
    1. Download Ollama from https://ollama.ai
    2. Run: ollama pull mistral
    3. Run: ollama serve
    
    Popular free models:
    - mistral: Fast, high-quality reasoning (recommended)
    - llama2: General purpose, good quality
    - neural-chat: Optimized for conversations
    - dolphin-mixtral: Excellent reasoning abilities
    """
    
    try:
        return ChatOllama(
            model=model_name,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,  # Low temperature for accuracy and consistency
        )
    except Exception as e:
        raise RuntimeError(
            f"Could not connect to Ollama on {OLLAMA_BASE_URL}. "
            f"Please ensure Ollama is running:\n"
            f"1. Download from https://ollama.ai\n"
            f"2. Run: ollama pull {model_name}\n"
            f"3. Run: ollama serve"
        ) from e


# Initialize with default model
try:
    selected_model = st.sidebar.selectbox(
        "Select Ollama Model",
        OLLAMA_MODELS,
        index=1,  # Default to mistral
        help="Select which free open-source model to use. "
             "Must be downloaded via 'ollama pull <model-name>'",
    )
    
    llm = initialize_llm(selected_model)
    st.sidebar.info(f"Selected model: {selected_model}")

except Exception as error:
    st.sidebar.error(f"⚠️ Unable to connect to Ollama: {error}")
    st.warning(
        "**Ollama not running!** Follow these steps:\n\n"
        "1. **Download Ollama** from https://ollama.ai\n"
        "2. **Pull a model** by running in terminal:\n"
        "   ```\n"
        "   ollama pull mistral\n"
        "   ```\n"
        "3. **Start Ollama** by running:\n"
        "   ```\n"
        "   ollama serve\n"
        "   ```\n"
        "4. **Refresh** this page once Ollama is running"
    )
    st.stop()


# ============================================================
# 3. CONSTANTS AND SCORING CONFIGURATION
# ============================================================

# These values are workshop assumptions.
# They should be validated with Marketing and Sales before production use.

HIGH_VALUE_BUSINESS_TYPES = {
    "Hotel",
    "Restaurant",
    "Catering",
    "Office",
    "Healthcare",
    "Education",
}

RELEVANT_CONTACT_LEVELS = {
    "Owner or General Manager": 15,
    "Procurement or Purchasing": 15,
    "Food and Beverage Management": 15,
    "Operations Management": 12,
    "Facilities or Workplace Management": 12,
    "Marketing": 8,
    "Other": 5,
}

PURCHASE_TIMEFRAME_POINTS = {
    "Immediate": 15,
    "Within 3 months": 15,
    "Within 3 to 6 months": 10,
    "More than 6 months": 5,
    "Not yet defined": 0,
}

REQUIRED_FIELDS = [
    "company_name",
    "country",
    "business_type",
    "contact_name",
    "contact_role",
    "product_interest",
    "purchase_timeframe",
]

STRICT_CONTEXT = """
Use only the information explicitly provided in the submitted lead form.

Do not use assumed or remembered knowledge about the company.
Do not browse for additional information.
Do not invent company size, locations, consumers, employees, consumption,
sales potential, existing contracts, installed equipment, lead intent,
or purchasing authority.

Do not recommend specific products unless a verified
product catalogue is provided in the input.

When information is missing, clearly identify it as missing.
Return valid JSON only, without Markdown or explanatory text.
"""


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def safe_json_dumps(value: Any) -> str:
    """
    Convert a Python object to a formatted JSON string.
    """

    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def normalize_text(value: Any) -> str:
    """
    Normalize text for basic rule matching.
    """

    if value is None:
        return ""

    return str(value).strip().lower()


def has_value(value: Any) -> bool:
    """
    Determine whether a form field has a meaningful value.
    """

    if value is None:
        return False

    if isinstance(value, str):
        return bool(value.strip())

    if isinstance(value, list):
        return len(value) > 0

    if isinstance(value, (int, float)):
        return value > 0

    return bool(value)


def prepare_lead_for_prompt(lead: Dict[str, Any]) -> Dict[str, str]:
    """
    Serialize the lead before injecting it into a LangChain prompt.
    """

    return {
        "lead": safe_json_dumps(lead)
    }


def flatten_missing_information(
    business_analysis: Dict[str, Any],
    demand_analysis: Dict[str, Any],
) -> List[str]:
    """
    Combine missing-information lists and remove duplicates.
    """

    combined = []

    for item in business_analysis.get("missing_information", []):
        if item and item not in combined:
            combined.append(item)

    for item in demand_analysis.get("missing_information", []):
        if item and item not in combined:
            combined.append(item)

    return combined


# ============================================================
# 5. LANGCHAIN PROMPTS
# ============================================================

business_fit_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            STRICT_CONTEXT
            + """
You are supporting the initial qualification of a business lead
for Out of Home Business product company.

Assess only the following dimensions:

1. Whether the stated business type appears relevant for a
   business-to-business food or beverage solution.
2. Whether the contact role appears relevant to a purchasing,
   operational, or food and beverage decision.
3. Whether the lead has provided a clear business requirement.
4. Which important qualification information is missing.

Do not calculate a numeric score.

Return exactly this JSON structure:

{{
  "business_type_fit": "high|medium|low|unknown",
  "contact_relevance": "high|medium|low|unknown",
  "requirement_clarity": "high|medium|low|unknown",
  "positive_signals": [
    "Signal explicitly supported by the form"
  ],
  "risk_signals": [
    "Risk explicitly supported by the form"
  ],
  "missing_information": [
    "Information required to qualify the lead"
  ],
  "evidence": [
    "Exact fact or close paraphrase from the form"
  ]
}}
""",
        ),
        (
            "user",
            "Submitted lead form:\n{lead}",
        ),
    ]
)


demand_analysis_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            STRICT_CONTEXT
            + """
Assess the potential demand described in the lead form.

Rules:

1. Identify whether the interest relates to beverages, food, both,
   or is unknown.
2. Use the submitted estimated daily demand if it is available.
3. Do not invent a number of consumers.
4. Do not invent a daily cup or food quantity.
5. If both the number of consumers and expected consumption rate
   are explicitly available, you may calculate a range.
6. Show the calculation used.
7. If there is insufficient information, return null for the range.
10. Do not name another specific product without a verified catalogue.
11. You may identify a general solution category, such as:
    professional beverage solution, coffee solution, food solution,
    vending solution, or manual review required.

Do not calculate a lead qualification score.

Return exactly this JSON structure:

{{
  "interest_category": "beverage|food|both|unknown",
  "general_solution_category": "string or null",
  "estimated_daily_beverage_cups": {{
    "minimum": null,
    "maximum": null,
    "calculation": null,
    "source": "lead_form|calculated_from_form|insufficient_information"
  }},
  "estimated_daily_food_portions": {{
    "minimum": null,
    "maximum": null,
    "calculation": null,
    "source": "lead_form|calculated_from_form|insufficient_information"
  }},
  "positive_signals": [
    "Signal explicitly supported by the form"
  ],
  "missing_information": [
    "Information required to assess demand"
  ],
  "evidence": [
    "Exact fact or close paraphrase from the form"
  ]
}}
""",
        ),
        (
            "user",
            "Submitted lead form:\n{lead}",
        ),
    ]
)


# ============================================================
# 6. INDIVIDUAL LANGCHAIN CHAINS
# ============================================================

business_fit_chain = (
    RunnableLambda(prepare_lead_for_prompt)
    | business_fit_prompt
    | llm
    | JsonOutputParser()
)

demand_analysis_chain = (
    RunnableLambda(prepare_lead_for_prompt)
    | demand_analysis_prompt
    | llm
    | JsonOutputParser()
)


# ============================================================
# 7. PARALLEL ANALYSIS
# ============================================================

parallel_analysis_chain = RunnableParallel(
    {
        "business_analysis": business_fit_chain,
        "demand_analysis": demand_analysis_chain,
        "lead": RunnablePassthrough(),
    }
)


# ============================================================
# 8. DETERMINISTIC LEAD SCORING
# ============================================================

def calculate_lead_score(
    analysis: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calculate a transparent lead score using deterministic rules.

    The LLM does not calculate or modify the score.
    """

    lead = analysis["lead"]
    business_analysis = analysis["business_analysis"]
    demand_analysis = analysis["demand_analysis"]

    total_score = 0
    score_breakdown = []

    # --------------------------------------------------------
    # A. BUSINESS TYPE FIT: MAXIMUM 25 POINTS
    # --------------------------------------------------------

    business_type = lead.get("business_type")

    if business_type in HIGH_VALUE_BUSINESS_TYPES:
        business_fit_points = 25
        business_fit_reason = (
            f"{business_type} is included in the configured "
            "target business types."
        )
    elif business_type == "Other":
        business_fit_points = 10
        business_fit_reason = (
            "The business type requires manual validation."
        )
    else:
        business_fit_points = 0
        business_fit_reason = (
            "No eligible business type was provided."
        )

    total_score += business_fit_points

    score_breakdown.append(
        {
            "criterion": "Business type fit",
            "points": business_fit_points,
            "maximum": 25,
            "reason": business_fit_reason,
        }
    )

    # --------------------------------------------------------
    # B. CONTACT RELEVANCE: MAXIMUM 15 POINTS
    # --------------------------------------------------------

    contact_role_category = lead.get("contact_role_category", "Other")

    contact_points = RELEVANT_CONTACT_LEVELS.get(
        contact_role_category,
        5,
    )

    total_score += contact_points

    score_breakdown.append(
        {
            "criterion": "Contact relevance",
            "points": contact_points,
            "maximum": 15,
            "reason": (
                f"Contact category selected: "
                f"{contact_role_category}."
            ),
        }
    )

    # --------------------------------------------------------
    # C. DEMAND POTENTIAL: MAXIMUM 30 POINTS
    # --------------------------------------------------------

    product_interest = lead.get("product_interest", [])
    estimated_daily_demand = lead.get(
        "estimated_daily_demand",
        0,
    )

    if not isinstance(estimated_daily_demand, (int, float)):
        estimated_daily_demand = 0

    if estimated_daily_demand >= 50:
        demand_points = 30
        demand_reason = (
            "The submitted estimated demand is at least "
            "50 cups or portions per day."
        )
    elif estimated_daily_demand >= 15:
        demand_points = 20
        demand_reason = (
            "The submitted estimated demand is between "
            "15 and 49 cups or portions per day."
        )
    elif estimated_daily_demand > 0:
        demand_points = 5
        demand_reason = (
            "The submitted estimated demand is below "
            "15 cups or portions per day."
        )
    else:
        demand_points = 0
        demand_reason = (
            "No estimated daily demand was provided."
        )

    total_score += demand_points

    score_breakdown.append(
        {
            "criterion": "Demand potential",
            "points": demand_points,
            "maximum": 30,
            "reason": demand_reason,
            "submitted_interest": product_interest,
        }
    )

    # --------------------------------------------------------
    # D. PURCHASE TIMEFRAME: MAXIMUM 15 POINTS
    # --------------------------------------------------------

    purchase_timeframe = lead.get(
        "purchase_timeframe",
        "Not yet defined",
    )

    timeframe_points = PURCHASE_TIMEFRAME_POINTS.get(
        purchase_timeframe,
        0,
    )

    total_score += timeframe_points

    score_breakdown.append(
        {
            "criterion": "Purchase timeframe",
            "points": timeframe_points,
            "maximum": 15,
            "reason": (
                f"Purchase timeframe selected: "
                f"{purchase_timeframe}."
            ),
        }
    )

    # --------------------------------------------------------
    # E. DATA COMPLETENESS: MAXIMUM 10 POINTS
    # --------------------------------------------------------

    completed_fields = sum(
        1
        for field in REQUIRED_FIELDS
        if has_value(lead.get(field))
    )

    completeness_ratio = completed_fields / len(REQUIRED_FIELDS)
    completeness_points = round(completeness_ratio * 10)

    total_score += completeness_points

    missing_form_fields = [
        field
        for field in REQUIRED_FIELDS
        if not has_value(lead.get(field))
    ]

    score_breakdown.append(
        {
            "criterion": "Form completeness",
            "points": completeness_points,
            "maximum": 10,
            "reason": (
                f"{completed_fields} of "
                f"{len(REQUIRED_FIELDS)} required fields completed."
            ),
            "missing_fields": missing_form_fields,
        }
    )

    # --------------------------------------------------------
    # F. REQUIREMENT CLARITY: MAXIMUM 5 POINTS
    # --------------------------------------------------------

    business_need = normalize_text(
        lead.get("business_need")
    )

    if len(business_need) >= 80:
        clarity_points = 5
        clarity_reason = (
            "A detailed business requirement was provided."
        )
    elif len(business_need) >= 20:
        clarity_points = 3
        clarity_reason = (
            "A short business requirement was provided."
        )
    else:
        clarity_points = 0
        clarity_reason = (
            "The business requirement is missing or too limited."
        )

    total_score += clarity_points

    score_breakdown.append(
        {
            "criterion": "Requirement clarity",
            "points": clarity_points,
            "maximum": 5,
            "reason": clarity_reason,
        }
    )

    # --------------------------------------------------------
    # G. QUALIFICATION STATUS
    # --------------------------------------------------------

    all_missing_information = flatten_missing_information(
        business_analysis=business_analysis,
        demand_analysis=demand_analysis,
    )

    # A lead with critical missing information should not be
    # automatically sent to Sales, even if other fields score well.

    critical_information_missing = (
        not has_value(lead.get("company_name"))
        or not has_value(lead.get("country"))
        or not has_value(lead.get("product_interest"))
    )

    if total_score >= 75 and not critical_information_missing:
        qualification_status = "Sales-qualified lead"
        recommended_route = "Sales follow-up"

    elif total_score >= 50:
        qualification_status = "Marketing-qualified lead"
        recommended_route = "Manual review"

    else:
        qualification_status = (
            "Additional information or nurturing required"
        )
        recommended_route = "Marketing nurture"

    analysis["score"] = total_score
    analysis["maximum_score"] = 100
    analysis["score_breakdown"] = score_breakdown
    analysis["qualification_status"] = qualification_status
    analysis["recommended_route"] = recommended_route
    analysis["missing_information"] = all_missing_information
    analysis["critical_information_missing"] = (
        critical_information_missing
    )

    return analysis


scored_analysis_chain = (
    parallel_analysis_chain
    | RunnableLambda(calculate_lead_score)
)


# ============================================================
# 9. FINAL SYNTHESIS PROMPT
# ============================================================

final_synthesis_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            STRICT_CONTEXT
            + """
Prepare an explainable lead qualification response using the supplied
lead form, parallel AI analyses, and deterministic score.

Critical rules:

1. The numeric score has already been calculated.
2. Do not change or recalculate the score.
3. Do not change the qualification status.
4. Do not change the recommended route.
5. Do not invent company or customer information.
6. Do not present assumptions as facts.
7. Do not name specific products unless they exist in a verified
   product catalogue included in the input.
8. Do not automatically reject a lead only because information
   is missing.
9. When information is missing, recommend clarification or nurturing.
10. The customer message must be professional and customer-friendly.
11. Do not expose internal scoring rules in the customer message.
12. Do not mention that the customer was rejected.
13. Do not mention NESGPT, LangChain, prompts, scores, or AI in the
    customer-facing message.
14. The sales summary may mention the score.
15. Base each positive signal on an explicit form value.

Return exactly this JSON structure:

{{
  "score": 0,
  "maximum_score": 100,
  "qualification_status": "",
  "recommended_route": "",
  "executive_summary": "",
  "positive_signals": [
    "Evidence-supported signal"
  ],
  "risk_signals": [
    "Evidence-supported risk or uncertainty"
  ],
  "missing_information": [
    "Information needed"
  ],
  "recommended_next_action": "",
  "questions_for_the_lead": [
    "Question that would improve qualification"
  ],
  "sales_representative_brief": "",
  "customer_message": ""
}}
""",
        ),
        (
            "user",
            "Scored lead analysis:\n{scored_analysis}",
        ),
    ]
)


def prepare_final_synthesis_input(
    scored_analysis: Dict[str, Any],
) -> Dict[str, str]:
    """
    Serialize the complete scored analysis for the final LLM call.
    """

    return {
        "scored_analysis": safe_json_dumps(scored_analysis)
    }


final_synthesis_chain = (
    RunnableLambda(prepare_final_synthesis_input)
    | final_synthesis_prompt
    | llm
    | JsonOutputParser()
)


# ============================================================
# 10. COMPLETE PIPELINE
# ============================================================

complete_lead_pipeline = (
    scored_analysis_chain
    | final_synthesis_chain
)


# ============================================================
# 11. FORM VALIDATION
# ============================================================

def validate_lead_form(
    lead: Dict[str, Any],
) -> List[str]:
    """
    Validate essential form fields before invoking the LLM.
    """

    errors = []

    if not has_value(lead.get("company_name")):
        errors.append("Company name is required.")

    if not has_value(lead.get("country")):
        errors.append("Country is required.")

    if not has_value(lead.get("contact_name")):
        errors.append("Contact name is required.")

    if not has_value(lead.get("email")):
        errors.append("Business email is required.")

    email = normalize_text(lead.get("email"))

    if email and ("@" not in email or "." not in email.split("@")[-1]):
        errors.append("Please provide a valid business email.")

    if not has_value(lead.get("product_interest")):
        errors.append(
            "At least one area of interest must be selected."
        )

    if not lead.get("contact_consent"):
        errors.append(
            "Consent to be contacted is required."
        )

    return errors


# ============================================================
# 12. STREAMLIT LEAD FORM
# ============================================================

st.divider()

st.subheader("Lead information")

with st.form("lead_qualification_form"):

    company_column, country_column = st.columns(2)

    with company_column:
        company_name = st.text_input(
            "Company name *",
            placeholder="Example: Alpine Business Hotel",
        )

    with country_column:
        country = st.text_input(
            "Country *",
            placeholder="Example: Switzerland",
        )

    contact_column, email_column = st.columns(2)

    with contact_column:
        contact_name = st.text_input(
            "Contact name *",
            placeholder="Example: Maria Example",
        )

    with email_column:
        email = st.text_input(
            "Business email *",
            placeholder="maria@example-company.com",
        )

    role_column, role_category_column = st.columns(2)

    with role_column:
        contact_role = st.text_input(
            "Job title *",
            placeholder="Example: Food and Beverage Manager",
        )

    with role_category_column:
        contact_role_category = st.selectbox(
            "Role category",
            [
                "Owner or General Manager",
                "Procurement or Purchasing",
                "Food and Beverage Management",
                "Operations Management",
                "Facilities or Workplace Management",
                "Marketing",
                "Other",
            ],
        )

    business_column, locations_column = st.columns(2)

    with business_column:
        business_type = st.selectbox(
            "Business type *",
            [
                "Hotel",
                "Restaurant",
                "Catering",
                "Office",
                "Healthcare",
                "Education",
                "Retail",
                "Other",
            ],
        )

    with locations_column:
        number_of_locations = st.number_input(
            "Number of locations",
            min_value=0,
            step=1,
            help=(
                "Enter zero if the number of locations is unknown."
            ),
        )

    consumers_column, demand_column = st.columns(2)

    with consumers_column:
        estimated_consumers = st.number_input(
            "Estimated consumers per day",
            min_value=0,
            step=1,
            help=(
                "Enter zero if the number of consumers is unknown."
            ),
        )

    with demand_column:
        estimated_daily_demand = st.number_input(
            "Estimated cups or food portions per day",
            min_value=0,
            step=1,
            help=(
                "Enter the lead's own estimate. "
                "Enter zero if it is unknown."
            ),
        )

    product_interest = st.multiselect(
        "Area of interest *",
        [
            "Coffee and hot beverages",
            "Other beverages",
            "Food",
        ],
    )

    current_solution = st.text_area(
        "Current solution",
        placeholder=(
            "Example: Current supplier, installed equipment, "
            "manual service, or no existing solution."
        ),
    )

    business_need = st.text_area(
        "Business requirement",
        placeholder=(
            "Describe what the company is looking for, "
            "the intended users, expected demand, and key requirements."
        ),
        height=120,
    )

    purchase_timeframe = st.selectbox(
        "Expected purchase timeframe *",
        [
            "Immediate",
            "Within 3 months",
            "Within 3 to 6 months",
            "More than 6 months",
            "Not yet defined",
        ],
    )

    contact_consent = st.checkbox(
        "I agree to be contacted regarding this business request. *"
    )

    submitted = st.form_submit_button(
        "Assess the lead",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# 13. EXECUTE PIPELINE
# ============================================================

if submitted:

    lead = {
        "company_name": company_name.strip(),
        "country": country.strip(),
        "contact_name": contact_name.strip(),
        "email": email.strip(),
        "contact_role": contact_role.strip(),
        "contact_role_category": contact_role_category,
        "business_type": business_type,
        "number_of_locations": number_of_locations,
        "estimated_consumers": estimated_consumers,
        "estimated_daily_demand": estimated_daily_demand,
        "product_interest": product_interest,
        "current_solution": current_solution.strip(),
        "business_need": business_need.strip(),
        "purchase_timeframe": purchase_timeframe,
        "contact_consent": contact_consent,
    }

    validation_errors = validate_lead_form(lead)

    if validation_errors:

        st.error(
            "The form cannot be assessed yet. "
            "Please correct the following information:"
        )

        for validation_error in validation_errors:
            st.write(f"• {validation_error}")

    else:

        try:
            with st.spinner("Assessing the lead..."):
                lead_result = complete_lead_pipeline.invoke(lead)

            st.session_state["latest_lead"] = lead
            st.session_state["latest_result"] = lead_result

        except Exception as error:
            st.error(
                "The lead assessment could not be completed."
            )

            with st.expander("Technical error details"):
                st.code(str(error))


# ============================================================
# 14. DISPLAY RESULT
# ============================================================

if "latest_result" in st.session_state:

    result = st.session_state["latest_result"]
    original_lead = st.session_state["latest_lead"]

    st.divider()
    st.subheader("Qualification result")

    metric_column, status_column, route_column = st.columns(3)

    with metric_column:
        st.metric(
            "Qualification score",
            f"{result.get('score', 0)}/"
            f"{result.get('maximum_score', 100)}",
        )

    with status_column:
        st.metric(
            "Qualification status",
            result.get(
                "qualification_status",
                "Not available",
            ),
        )

    with route_column:
        st.metric(
            "Recommended route",
            result.get(
                "recommended_route",
                "Not available",
            ),
        )

    st.info(
        result.get(
            "executive_summary",
            "No executive summary was generated.",
        )
    )

    left_column, right_column = st.columns(2)

    with left_column:

        st.markdown("#### Positive signals")

        positive_signals = result.get(
            "positive_signals",
            [],
        )

        if positive_signals:
            for signal in positive_signals:
                st.write(f"✅ {signal}")
        else:
            st.write("No positive signals identified.")

        st.markdown("#### Missing information")

        missing_information = result.get(
            "missing_information",
            [],
        )

        if missing_information:
            for item in missing_information:
                st.write(f"❓ {item}")
        else:
            st.write("No important missing information identified.")

    with right_column:

        st.markdown("#### Risks and uncertainties")

        risk_signals = result.get(
            "risk_signals",
            [],
        )

        if risk_signals:
            for risk in risk_signals:
                st.write(f"⚠️ {risk}")
        else:
            st.write("No specific risk signals identified.")

        st.markdown("#### Recommended next action")

        st.success(
            result.get(
                "recommended_next_action",
                "Manual review required.",
            )
        )

    st.markdown("#### Questions for the lead")

    questions = result.get(
        "questions_for_the_lead",
        [],
    )

    if questions:
        for question_number, question in enumerate(
            questions,
            start=1,
        ):
            st.write(f"{question_number}. {question}")
    else:
        st.write("No additional questions generated.")

    st.markdown("#### Brief for the sales representative")

    st.write(
        result.get(
            "sales_representative_brief",
            "No sales brief generated.",
        )
    )

    st.markdown("#### Proposed customer message")

    st.text_area(
        "Customer-facing message",
        value=result.get(
            "customer_message",
            "",
        ),
        height=180,
    )

    with st.expander("View submitted lead data"):
        st.json(original_lead)

    with st.expander("View complete AI result"):
        st.json(result)