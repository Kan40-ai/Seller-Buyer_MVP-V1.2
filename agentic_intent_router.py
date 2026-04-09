import json
# import google.generativeai as genai
from groq import Groq

class StageZeroRouter:
    def __init__(self, llm_client):
        # Pass your Gemini Flash client here
        self.llm = llm_client 

    def classify_intent(self, user_query: str, persona: str = None, 
                    persona_label: str = None) -> str:

        # Persona context line — injected into prompt
        persona_context = ""
        if persona and persona_label:
            persona_context = (
                f"\nThe user is logged in as a {persona_label}. "
                f"Use this as additional context when the query is ambiguous — "
                f"e.g. a Loan Officer asking 'run this' most likely means LOAN_BOXING, "
                f"a Buyer asking 'find matches' most likely means MATCHING_ENGINE."
            )


        prompt = f"""
        You are an intelligent Intent Classification Agent for "Portfolio Co-Pilot."
        {persona_context}
        
        CATEGORIES:
        [MATCHING_ENGINE] - **ACTION ORIENTED**. 
        Trigger this ONLY if the user wants to PERFORM a counterparty match or EXECUTE a 5300 analysis. 
        Keywords:"run matching", "find me a buyer", "process this 5300", "execute matching report",
        "find me potential buyers", "identify top sellers", "find counterparty", "opposite party".
        CRITICAL: Use this if a 5300 Excel file is uploaded and the user says "process this."
        CRITICAL: If user says "explain the matching", "explain the scoring", 
        "explain the logic behind", "explain the above", "explain these results" 
        → route to MATCHING_ENGINE not KNOWLEDGE_BASE.
        These are requests to explain a prior engine run, not general concepts.
        "CRITICAL: If user mentions a specific CU ID like 'CU_COMMUNITY_003', "
        "'CU_ACU_001', 'CU_ALAMO_004' or any 'CU_' prefixed name → "
        "route to MATCHING_ENGINE, NOT MARKET_DATA or KNOWLEDGE_BASE. "
        "These are counterparty-specific questions about a prior engine run."

        [LOAN_BOXING] - ACTION ORIENTED.
        Trigger this if the user wants to EXECUTE, EXPLAIN, FILTER, or REPORT
        on a loan tape boxing pipeline.

        Keywords:
        - Run/Execute: "run loan boxing", "clean this portfolio", "generate box assignments",
        "process this loan tape", "run boxing with standard filters",
        "apply custom filters before boxing", "prepare this tape for sale"
        - Download/Export: "download the excel", "export the results",
        "give me the loan boxing file", "download box assignment"
        - Report: "generate executive report", "prepare the portfolio report",
        "executive summary report", "phase 3 report", "final portfolio pdf"
        - Explain: "how does the loan boxing work", "explain the box assignment logic",
        "what is the risk banding methodology", "explain the boxing engine"
        - Filter refinement: "apply FICO filter", "refine the portfolio",
        "exclude DPD loans", "filter by LTV", "tighten the filters"
        - What-if: "what if I tighten FICO to 720", "how many loans survive if LTV max 80%"
        - Re-run: "run again with this tape", "use this new file", "start over"

        CRITICAL: Use this if a Loan Tape (CSV/Excel) is uploaded and the user
        wants to run, filter, explain, download, or report on box assignments.
        CRITICAL: Also trigger if chatbot_boxing_phase is active in session —
        user is continuing a multi-phase boxing workflow regardless of intent.
        CRITICAL: Do NOT trigger for 5300 Call Reports — those go to MATCHING_ENGINE.

        [KNOWLEDGE_BASE] - **CONCEPTUAL/INFO ORIENTED**. 
        Trigger this for all "What is", "How does", or "Why" questions.
        Example: "What is loan boxing?", "How do you calculate the LTD bonus?", "Explain risk-weighting."
        If the user is asking about the METHODOLOGY of boxing without asking to actually PROCESS a file, route here.

        [GENERAL_ASSISTANCE] - Greetings, system troubleshooting, or off-topic queries.

        [DOCUMENT_QA] - **EXTRACTION ORIENTED**.
        Trigger this if the user wants to extract, find, or read specific values,
        text, or information FROM an uploaded document.
        Keywords: "from attached", "from this report", "from the document",
        "find value", "what does it say", "extract", "what is the value of",
        "from attached report", "in this document", "according to this".
        CRITICAL: If user says "find X from attached/this report/document" → route here,
        NOT to MATCHING_ENGINE.

        [MARKET_DATA] - BENCHMARKING ORIENTED.
        Trigger if the user wants to benchmark a CU against peers or compare against market/industry data.
        Keywords: "benchmark", "compare to peers", "peer group", "industry average",
        "how does this CU compare", "market data", "live data", "peer comparison",
        "how do we rank", "where do we stand", "industry benchmark", "NCUA average",
        "compared to similar CUs", "peer analysis".
        CRITICAL: Use this if user asks how a CU compares to others — NOT to run matching engine.

        
        ======================================================================
        ROUTING RULES:
        - EXECUTING/PROCESSING/RUNNING    → [MATCHING_ENGINE] or [LOAN_BOXING]
        - LOAN TAPE BOXING/FILTERING/REPORT → [LOAN_BOXING]
        - LEARNING/DEFINING/EXPLAINING    → [KNOWLEDGE_BASE]
        - EXTRACTING FROM A DOCUMENT      → [DOCUMENT_QA]
        - BENCHMARKING/PEER COMPARISON    → [MARKET_DATA]
        - GREETING/OFF-TOPIC              → [GENERAL_ASSISTANCE]
        ======================================================================

        User Query: "{user_query}"

        Return your answer as a JSON object in this exact format:
        {{
            "intent": "INTENT_NAME_HERE"
        }}
        """

        try:
            # Call Grok model with JSON enforcement
            response = self.llm.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )

            
            routing_data = json.loads(response.choices[0].message.content)
            intent = routing_data.get("intent", "GENERAL_ASSISTANCE")

            print(f"[Router] query='{user_query}' → intent={intent}")
            
            # Updated valid intents for this project
            valid_intents = ["MATCHING_ENGINE", "LOAN_BOXING", "KNOWLEDGE_BASE",
                        "DOCUMENT_QA", "GENERAL_ASSISTANCE", "MARKET_DATA"]
            if intent not in valid_intents:
                return "GENERAL_ASSISTANCE"
                
            return intent
            
        except Exception as e:
            print(f"⚠️ Router Error ({e}). Defaulting to GENERAL_ASSISTANCE.")
            return "GENERAL_ASSISTANCE"