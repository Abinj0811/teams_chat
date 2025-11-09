import os
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# import numpy as np

from dotenv import load_dotenv
load_dotenv()
import json
import random
import re
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from openai import OpenAI
from datetime import datetime
import os

# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# memory.save_context({"input": "Hi"}, {"output": "Hello there!"})
# print(memory.load_memory_variables({}))

# exit()

from datetime import datetime
from typing import List, Dict
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from openai import OpenAI
from langchain_classic.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from botbuilder.core import ActivityHandler, MessageFactory, TurnContext
from botbuilder.schema import ChannelAccount, ActivityTypes, Activity



class ThinkpalmCosmosRAGmethod2:
    def __init__(self, cosmos_endpoint, cosmos_key, db_name, container_name, history_container_name):
        # Cosmos setup
        # print(cosmos_endpoint, cosmos_key)

        self.client = CosmosClient(url=cosmos_endpoint, credential=cosmos_key)
        self.db = self.client.get_database_client(db_name)
        self.container = self.db.get_container_client(container_name)
        self.history_container = self.db.get_container_client(history_container_name)
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        # OpenAI API client
        self.llm = OpenAI(api_key=OPENAI_API_KEY)

        # In-memory chat memory store (per user)
        self.memory_store = {}
        self.top_k = 5
        self.last_question_cache = {}
        
        self.SMALL_TALK_RESPONSES = {
        "greeting": [
            "Hello! I am Thinkpalm's Corporate Knowledge Assistant. How may I be of assistance with your business query?",
            "Good day. Thank you for reaching out. I'm ready to help with any policy or knowledge questions you may have.",
            "Hi there. I trust you are having a productive day. Please let me know your question.",
            "Welcome! I am here to provide accurate and professional support. What information are you seeking?",
        ],
        "thanks": [
            "You are most welcome. Is there anything else I can clarify or retrieve for you?",
            "My pleasure. Do not hesitate to ask if further information is required.",
            "Glad to be of assistance. Have a productive day.",
        ],
        "who_are_you": [
            "I am Thinkpalm's Corporate Knowledge Assistant, designed to provide information and policy details from our internal knowledge base.",
        ],
        "generic_positive": [
            "That is kind of you to say. I am functioning optimally and ready to address your corporate queries.",
        ]
    }
    NOVATION_RULES = """
    If the question involves Novation, Amendment, or Cancellation:
    - Use the exact policy title as stated in the Document Context.
    - Include all Authorised Approvers, Deliberations, Reviews, and Co-Management Departments.
    - Mention "GPM HOD decides whether it is Important or Others" exactly if it appears.
    """

    COST_RULES = """
    If the question involves costs, fees, or IT-related budgets:
    - Treat each cost category (Implementation, Maintenance, etc.) separately.
    - Extract full approval lines for each.
    - **CRITICAL: When multiple amount thresholds apply, ALWAYS use the MOST SPECIFIC threshold that applies.**
      For example, if an amount is $8,300:
      - It qualifies for both "Less than US$50,000" AND "Less than US$25,000"
      - You MUST use the more specific "Less than US$25,000" rule, NOT the general "Less than US$50,000" rule
      - Always check if a more specific threshold exists before applying a general one
    - Present in format: (A) Category — [details]. (B) Category — [details].
    """

    SUBTYPE_RULES = """
    If the query refers to subtypes (e.g., CLI, FDD, DTH, TCL):
    - Identify specific responsible department for that subtype.
    - Prefer subtype-level rule over general rules.
    """

    COMMITTEE_RULES = """
    If the question involves a committee (e.g., Ship Management Committee, Budget Committee, etc.):
    - Include ALL structural information about the committee:
      * Chairperson (if specified)
      * All Members (including Executive Officers and sub-members)
      * Head of Department (Secretariat) - this is critical and must be included
      * Any other roles or responsibilities mentioned
    - Do NOT omit any structural role, even if the question only asks about "members"
    - Present the complete committee structure in your answer.
    """


    
    # ------------------------------------------------------------
    # MEMORY MANAGEMENT
    # ------------------------------------------------------------
    def get_memory(self, user_id):
        """Create or retrieve summary memory per user."""
        if user_id not in self.memory_store:
            self.memory_store[user_id] = ConversationSummaryMemory(
                llm=ChatOpenAI(model="gpt-4.1", temperature=0),
                return_messages=True
            )
        return self.memory_store[user_id]

    # ------------------------------------------------------------
    # COSMOS HELPERS
    # ------------------------------------------------------------
    def _ensure_database(self, db_name):
        try:
            return self.client.create_database_if_not_exists(id=db_name)
        except Exception as e:
            print(f"Error ensuring database: {e}")
            raise

    def _ensure_container(self, container_name, partition_key="id"):
        try:
            return self.db.create_container_if_not_exists(
                id=container_name, partition_key=PartitionKey(path=f"/{partition_key}")
            )
        except Exception as e:
            print(f"Error ensuring container '{container_name}': {e}")
            raise

    # ------------------------------------------------------------
    # CHAT HISTORY STORAGE (COSMOS)
    # ------------------------------------------------------------
    def get_user_history(self, user_id):
        try:
            query = f"""
            SELECT * FROM c 
            WHERE c.user_id = '{user_id}' 
            ORDER BY c.timestamp DESC OFFSET 0 LIMIT 5
            """
            items = list(self.history_container.query_items(query=query, enable_cross_partition_query=True))
            items.reverse()
            return items
        except exceptions.CosmosResourceNotFoundError:
            print("⚠️ Chat history container not found. Creating now.")
            self.history_container = self._ensure_container("chathistory", partition_key="user_id")
            return []
        except Exception as e:
            print(f"Error reading history: {e}")
            return []

    def save_chat_message(self, user_id, user_msg, assistant_msg):
        item = {
            "id": f"{user_id}-{datetime.utcnow().isoformat()}",
            "user_id": user_id,
            "user": user_msg,
            "assistant": assistant_msg,
            "timestamp": datetime.utcnow().isoformat()
        }
        try:
            self.history_container.upsert_item(item)
        except Exception as e:
            print(f"Error saving message: {e}")

    # ------------------------------------------------------------
    # RAG SEARCH
    # ------------------------------------------------------------
    def search_cosmos_documents(self, query_embedding):
        """
        Query Cosmos using the VectorDistance API. Ensure the embedding is JSON-serialized
        so Cosmos receives a correct array. Returns a list of docs with id, text, metadata, score.
        """
        # Use json.dumps to produce a properly formatted array literal
        embedding_str = json.dumps(query_embedding)

        query = f"""
        SELECT TOP {self.top_k}
            c.id, c.text, c.metadata,
            VectorDistance(c.vector_embedding, {embedding_str}) AS score
        FROM c
        ORDER BY VectorDistance(c.vector_embedding, {embedding_str})
        """
        try:
            results = list(self.container.query_items(query=query, enable_cross_partition_query=True))
        except Exception as e:
            print(f"Error running vector query: {e}")
            results = []

        docs = []
        for d in results:
            docs.append({
                "id": d.get("id"),
                "text": d.get("text", ""),
                "metadata": d.get("metadata", {}),
                "score": float(d.get("score", 0))
            })
        # debug
        print(f"[DEBUG] search_cosmos_documents returned {len(docs)} docs (top_k={self.top_k})")
        return docs

    def _add_retrieval_hints(self, text: str) -> str:
        """
        Adds structured retrieval hints (thresholds, IT vs Non-IT context, novation, golf, etc.)
        to bias vector search toward the correct policy section.
        """
        # Normalize USD amounts (e.g., "$15000" → "US$ 15,000")
        def _format_usd(n: int) -> str:
            s = f"{n:,}".replace(",", ",")
            return f"US$ {s}"

        normalized_hints = []

        # Extract first amount mention
        m = re.search(r"(?:US\$|USD|\$)\s*([0-9]{1,3}(?:[, ]?[0-9]{3})*|[0-9]+)", text, flags=re.IGNORECASE)
        if m:
            raw = m.group(1)
            num = int(re.sub(r"[^0-9]", "", raw)) if raw else None
            if num is not None:
                normalized_hints.append(_format_usd(num))
                # Bucket thresholds
                if num < 10000:
                    normalized_hints.append("Less than US$10,000")
                elif num < 25000:
                    normalized_hints.append("Less than US$25,000")
                elif num < 50000:
                    normalized_hints.append("Less than US$50,000")
                elif num >= 50000:
                    normalized_hints.append("US$50,000 or more")

                # IT context thresholds
                if re.search(r"\b(it|information technology|software|system|implementation|hardware)\b", text, re.I):
                    if num >= 50000:
                        normalized_hints.append("US$50,000 or more - IT assets")
                    elif num >= 25000:
                        normalized_hints.append("US$25,000 or more - IT assets")
                    elif num >= 10000:
                        normalized_hints.append("US$10,000 or more - IT assets")
                    else:
                        normalized_hints.append("Less than US$10,000 - IT assets")

        # Domain-specific cues
        if re.search(r"golf\s*course\s*membership", text, re.I):
            normalized_hints += ["Golf course membership", "Fixed assets", "Non-IT assets"]

        # Generic acquisition / implementation hints
        if re.search(r"\b(acquisition|purchase|approve|implement|implementation)\b.*\b(asset|software|system|equipment|technology)\b", text, re.I):
            normalized_hints += [
                "Acquisition of fixed assets",
                "IT-related fixed assets",
                "Non-IT fixed assets",
                "Software (including development cost)",
                "System implementation"
            ]

        # IT hints (highest priority)
        if re.search(r"\b(it|software|hardware|system|implementation)\b", text, re.I):
            it_hints = [
                "Acquisition, disposal of IT related assets",
                "IT-related assets",
                "Software (including development cost)",
                "Computer equipment",
                "IT equipment and fixtures"
            ]
            normalized_hints = it_hints + normalized_hints

        # Charter/Novation hints
        if re.search(r"\b(novation|charter|time charter|bare boat)\b", text, re.I):
            normalized_hints += [
                "Novation of the contract",
                "Amendment/ Cancellation of Time Charterer",
                "Time Charter",
                "Important",
                "Others"
            ]

        # Duration/tenor hints for Time Charter questions
        dur_match = re.search(r"\b(\d+)\s*[- ]?\s*year", text, flags=re.IGNORECASE)
        is_charter_in = re.search(r"\bcharter\s*in\b", text, flags=re.IGNORECASE) is not None
        is_charter_out = re.search(r"\bcharter\s*out\b", text, flags=re.IGNORECASE) is not None
        if dur_match and (is_charter_in or is_charter_out):
            years = int(dur_match.group(1))
            # Add canonical section headers to bias retrieval
            normalized_hints += ["Handling of Time Charter", "Conclusion/ Amendment/ Cancellation of Time Charterer", "Time Charter"]
            if is_charter_in:
                normalized_hints.append("Charter in")
                if years >= 5:
                    normalized_hints.append("5 years and more")
                elif years > 1:
                    normalized_hints.append("More than 1 year and less than 5 years")
                else:
                    normalized_hints.append("1 year and less")
            elif is_charter_out:
                normalized_hints.append("Charter out")
                if years >= 3:
                    normalized_hints.append("3 years and more")
                elif years > 1:
                    normalized_hints.append("More than 1 year and less than 3 years")
                else:
                    normalized_hints.append("1 year and less")

        # Service Agreement with MCTSPR subsidiaries hints (HIGH PRIORITY)
        subs_keys = ["mctwtn", "mcttky", "mctdbi", "mctldn", "mctrmd", "mcthou", "mctcph", "mctbog", "unix", "tms"]
        if re.search(r"\b(service agreement|conclusion|termination|revision)\b", text, re.I) and \
           any(re.search(rf"\b{key}\b", text, re.I) for key in subs_keys):
            # Prioritize service agreement with MCTSPR subsidiaries content
            normalized_hints = [
                "Conclusion / Termination / Revision of service agreement with MCTSPR subsidiaries",
                "Responsible department for conclusion / termination / revision of service agreement",
                "Service agreement with MCTSPR subsidiaries"
            ] + normalized_hints  # Put service agreement hints first

        # Deduplicate & attach hints
        if normalized_hints:
            unique_hints = []
            seen = set()
            for h in normalized_hints:
                if h not in seen:
                    unique_hints.append(h)
                    seen.add(h)
            text = text + "\n\nHINTS: " + "; ".join(unique_hints)

        return text

    def _extract_policy_section_keywords(self, question: str, q_lower: str) -> List[str]:
        """
        General method to extract canonical policy section titles/keywords from questions.
        Returns a list of policy section keywords that should be used for targeted fallback search.
        """
        keywords = []
        
        # Mapping of question patterns to policy section titles
        # Format: (pattern_keywords, required_terms, section_keywords)
        policy_mappings = [
            # Service Agreement
            (["service agreement"], ["mctwtn", "mcttky", "mctdbi", "mctldn", "mctrmd", "mcthou", "mctcph", "mctbog", "unix", "tms"], [
                "Conclusion/ Termination/ Revision of service agreement with MCTSPR subsidiaries",
                "Conclusion / Termination / Revision of service agreement with MCTSPR subsidiaries",
                "Responsible department for conclusion/ termination/ revision of service agreement",
                "Responsible department for conclusion / termination / revision of service agreement"
            ]),
            # Novation
            (["novation"], [], [
                "Novation of the contract",
                "Novation of the contract Important",
                "Novation of the contract Others"
            ]),
            # Insurance (P&I, CLI, FDD)
            (["p&i", "insurance", "cli", "fdd", "dth", "tcl"], [], [
                "Purchasing Insurance",
                "Vessel-related insurances",
                "Other insurances",
                "Annual plan",
                "Execution of annual plan"
            ]),
            # Time Charter
            (["time charter", "charter"], [], [
                "Handling of Time Charter",
                "Conclusion/ Amendment/ Cancellation of Time Charterer",
                "Time Charter"
            ])
        ]
        
        # Check deterministic patterns first
        for pattern_keywords, required_terms, section_keywords in policy_mappings:
            # Check if any pattern keyword matches
            if any(pk in q_lower for pk in pattern_keywords):
                # If required terms exist, check them too
                if required_terms:
                    if any(rt in q_lower for rt in required_terms):
                        keywords.extend(section_keywords)
                else:
                    keywords.extend(section_keywords)
        
        # Use LLM to extract additional policy section titles if no matches found
        if not keywords:
            try:
                extraction_prompt = f"""Extract the exact policy section title(s) from this question that would appear in an authority regulations document.

Examples:
- "Novation of Time Charter contract" → ["Novation of the contract"]
- "P&I Insurance (CLI/FDD) for Policy Year 2025" → ["Purchasing Insurance", "Vessel-related insurances", "Annual plan"]
- "Service agreement with MCTWTN" → ["Conclusion/ Termination/ Revision of service agreement with MCTSPR subsidiaries"]
- "Approval for golf membership" → ["Acquisition, disposal of assets", "Golf course membership"]

Return only the policy section titles as a comma-separated list, or "None" if no specific policy section can be identified.

Question: {question}

Policy section titles:"""
                
                resp = self.llm.chat.completions.create(
                    model="gpt-4.1",
                    messages=[{"role": "user", "content": extraction_prompt}],
                    temperature=0.0
                )
                extracted = resp.choices[0].message.content.strip()
                
                if extracted.lower() not in {"none", "n/a", ""}:
                    # Split by comma and clean up
                    extracted_keywords = [kw.strip() for kw in extracted.split(",") if kw.strip()]
                    keywords.extend(extracted_keywords)
            except Exception as e:
                print(f"[DEBUG] LLM policy section extraction error: {e}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                unique_keywords.append(kw)
        
        return unique_keywords[:10]  # Limit to top 10 to avoid too many queries

    def _save_retrieved_sections(self, question: str, query_for_retrieval: str, retrieved_docs: List[Dict], subtype: str, fallback_candidates: List[str]):
        """
        Save retrieved sections to a dedicated debug file for easier inspection.
        """
        debug_filename = "retrieved_sections_debug.txt"
        try:
            with open(debug_filename, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"TIMESTAMP: {datetime.utcnow().isoformat()}\n")
                f.write(f"{'='*80}\n\n")
                
                # Query information
                f.write(f"ORIGINAL QUESTION:\n{question}\n\n")
                f.write(f"REWRITTEN QUERY FOR RETRIEVAL:\n{query_for_retrieval}\n\n")
                f.write(f"EXTRACTED SUBTYPE: {subtype if subtype else 'None'}\n")
                f.write(f"FALLBACK CANDIDATES: {', '.join(fallback_candidates) if fallback_candidates else 'None'}\n\n")
                
                # Retrieved sections summary
                f.write(f"RETRIEVED SECTIONS COUNT: {len(retrieved_docs)}\n")
                f.write(f"{'-'*80}\n\n")
                
                # Each retrieved section
                for i, doc in enumerate(retrieved_docs, 1):
                    f.write(f"\n{'#'*80}\n")
                    f.write(f"RETRIEVED SECTION #{i}\n")
                    f.write(f"{'#'*80}\n")
                    f.write(f"Score: {doc.get('score', 0):.4f}\n")
                    f.write(f"Priority: {doc.get('priority', 0)}\n")
                    f.write(f"Document ID: {doc.get('id', 'N/A')}\n")
                    if doc.get('metadata'):
                        f.write(f"Metadata: {json.dumps(doc.get('metadata', {}), indent=2, ensure_ascii=False)}\n")
                    f.write(f"\n{'─'*80}\n")
                    f.write(f"CONTENT:\n")
                    f.write(f"{'─'*80}\n")
                    f.write(f"{doc.get('text', '')}\n")
                    f.write(f"\n{'─'*80}\n\n")
                
                f.write(f"\n{'='*80}\n")
                f.write(f"END OF RETRIEVAL DEBUG ENTRY\n")
                f.write(f"{'='*80}\n\n")
        except Exception as e:
            print(f"[DEBUG] Error saving retrieved sections to debug file: {e}")

    def _keyword_fallback_search(self, keyword):

        """

        If vector retrieval misses the exact table row, use a simple keyword substring search

        against the stored documents in Cosmos (or text index). This helps guarantee we get

        explicit lines like "Appointment/ Removal of Directors/ EOs".

        """

        # escape single quotes in keyword for SQL

        safe_kw = keyword.replace("'", "''")

        query = f"""

        SELECT TOP 20 c.id, c.text, c.metadata

        FROM c

        WHERE CONTAINS(c.text, '{safe_kw}')

        """

        try:

            results = list(self.container.query_items(query=query, enable_cross_partition_query=True))

        except Exception as e:

            print(f"Keyword fallback search error: {e}")

            results = []



        docs = []

        for d in results:

            docs.append({

                "id": d.get("id"),

                "text": d.get("text", ""),

                "metadata": d.get("metadata", {}),

                "score": 0.0

            })

        print(f"[DEBUG] _keyword_fallback_search('{keyword}') found {len(docs)} docs")

        return docs

    def _filter_by_policy_header(self, docs: List[Dict], must_include: str | None) -> List[Dict]:
        """
        Keep only docs containing the specified policy header text (case-insensitive).
        Falls back to original docs if filtering removes everything.
        """
        if not must_include:
            return docs
        key = must_include.lower()
        filtered = [d for d in docs if key in d.get("text", "").lower()]
        return filtered or docs


    @staticmethod
    def format_context(docs: List[Dict]) -> str:
        """
        Format retrieved docs for LLM prompt — removes debug doc labels so model doesn’t cite “Doc X”.
        """
        return "\n\n".join(d.get("text", "") for d in docs)

    # Add this new helper method to your class (or implement the logic inline)
    def _rewrite_query(self, memory_history: str, current_question: str) -> str:
        """Uses the LLM to convert an ambiguous follow-up into a standalone query."""
        
        # This prompt instructs the LLM to perform the rewriting task.
        rewrite_prompt = f"""
                You are a query rewriter for a Retrieval-Augmented Generation (RAG) system.

        Your goal is to rewrite the 'Current User Question' into a single, self-contained,
        and contextually complete search query using the 'Conversation History' to fill in
        missing references.

        CRITICAL: Prioritize the MAIN QUESTION FOCUS (what is being asked) over specific amounts or thresholds.
        - The main question focus (e.g., "approval and departments", "who approves", "what is the process") should be emphasized
        - Amounts (e.g., USD800,000, $50,000) are context/filters but should NOT override the main query focus
        - For questions like "What approval and departments are involved for X with amount Y?", prioritize "approval and departments for X" over just "amount Y"

        Strict Rules:
        1. Preserve the logical meaning, phrasing, and operators (e.g., "and", "or", "/") exactly as in the user's question. 
        - Do NOT replace "or" with "and", or vice versa.
        - Do NOT merge multiple conditions unless they are identical in meaning.
        2. Do NOT infer or generalize beyond the user's wording — stay faithful to the intent.
        3. Include relevant context from Conversation History only if it clarifies **what** the user is referring to.
        4. Emphasize the main question focus (what is being asked) while keeping amounts as supporting context.
        5. Output must be a single concise query — no explanations, no extra text.

        Example:
        History:
        User: What are the benefits of the new HR policy?
        Assistant: The policy provides flexible PTO and a stipend.
        Current User Question: What is the stipend amount?
        Rewritten Query: What is the stipend amount for the new HR policy?

        Example with amount:
        Current User Question: What approval and departments are involved for the conclusion of service agreement with MCTWTN for admin cost sharing of USD800,000?
        Rewritten Query: What approval and departments are involved for the conclusion of service agreement with MCTWTN (admin cost sharing USD800,000)

        Conversation History:
        {memory_history}

        Current User Question:
        {current_question}
        Rewritten Query:
        """
        
        response = self.llm.chat.completions.create(
            model="gpt-4.1", # Use a fast, inexpensive model for this step
            messages=[{"role": "user", "content": rewrite_prompt}],
            temperature=0.0 # Set low temperature for factual rewriting
        )
        return response.choices[0].message.content.strip()

   
    # ------------------------------------------------------------
    # ASK METHOD (MAIN PIPELINE)
    # ------------------------------------------------------------
    # New (Simplified) ask method structure:
    # def _extract_keywords(self, question: str) -> str | None:
    #     """
    #     Extract the exact canonical policy title for lookup.
    #     Adds hard mapping for IT/software-related terms.
    #     """
    #     IT_POLICY_TITLE = (
    #         "Acquisition, disposal of IT related assets "
    #         "a) Equipment and fixtures b) Intellectual property "
    #         "c) Software (including development cost) d) Other IT assets"
    #     )

    #     q = (question or "").lower()

    #     # 🔒 HARD MAP — ensures software/IT questions use the correct policy title
    #     if any(w in q for w in [
    #         "software", "it system", "implementation", "maintenance",
    #         "development cost", "it-related", "it related", "application system",
    #         "system upgrade", "license", "it solution"
    #     ]):
    #         return IT_POLICY_TITLE

    #     # Otherwise, fallback to LLM-based extraction
    #     extraction_prompt = f"""
    #     You are a policy title extractor. Your task is to identify the single, canonical, and EXACT POLICY TITLE 
    #     corresponding to the user's question, which must be used for a literal database lookup.

    #     The extracted title MUST be a complete, literal match for the policy in question.

    #     If the question is general (e.g., 'What is the cost?', 'How are you?'), return 'None'.

    #     Examples:
    #     - User: Who approves the Appointment of new directors? -> Output: Appointment/ Removal of Directors/ EOs
    #     - User: What is the process for paying a cancellation fee? -> Output: Payment of cancellation fee/ penalty charge
    #     - User: What is the travel policy? -> Output: Business Travel
    #     - User: What is the cost? -> Output: None

    #     User Question:
    #     {question}

    #     Extracted Keyword (or 'None'):
    #     """
    #     try:
    #         response = self.llm.chat.completions.create(
    #             model="gpt-4.1",
    #             messages=[{"role": "user", "content": extraction_prompt}],
    #             temperature=0.0
    #         )
    #         keyword = response.choices[0].message.content.strip()
    #         keyword = keyword.replace('"', '').replace("'", '').split('\n')[0].strip()
    #         return keyword if keyword.lower() != 'none' and keyword else None
    #     except Exception:
    #         return None


    
# Define these as class attributes in your Knowledge Assistant class
# or as constants accessible by the method.
    
    def _is_small_talk(self, question: str) -> bool:
        """
        Classifies a question as small talk using deterministic rules.
        This version is strict, requiring the small talk phrase to dominate the query.
        """
        
        question_lower = question.lower().strip()
        question_words = question_lower.split()
        
        # 1. Define common small talk keywords
        GREETINGS = ["hello",'hloo', "hi", "hey", "good morning", "good evening", "greetings"]
        INQUIRIES = ["how are you", "what's up", "what are you doing", "who are you"]
        AFFIRMATIONS = ["thank you", "thanks", "i appreciate it", "bye", "goodbye"]
        
        # Consolidate phrases, including squashed versions (e.g., "thankyou")
        all_phrases = GREETINGS + AFFIRMATIONS + INQUIRIES
        squashed_phrases = [p.replace(' ', '').replace("'", "") for p in all_phrases if ' ' in p]
        all_checks = all_phrases + squashed_phrases
        
        # 2. Iteratively check for substring matches with strict dominance rules
        for phrase in all_checks:
            if phrase in question_lower:
                phrase_len = len(phrase.split())
                question_len = len(question_words)
                
                # A. Exact Match (The most definitive check)
                if phrase == question_lower:
                    return True
                
                # B. Dominance Check: The small talk phrase is nearly the entire query (e.g., 1-2 extra words)
                # Example: "Hi there" (2 words) or "Thank you so much" (4 words)
                if question_len <= phrase_len + 2:
                    return True

                # C. Boundary Check for leading/trailing small talk
                # This catches "Hi, can you tell me the policy?" only if the policy part is also short (max 5 words)
                if (question_lower.startswith(phrase) or question_lower.endswith(phrase)) and question_len <= 5:
                    return True
                    
        # 3. Heuristic: Final catch for very short, non-standard simple queries (e.g., "Thanks")
        if len(question_words) <= 2 and any(word in question_lower for word in GREETINGS + ["thanks", "bye"]):
            return True
        
        return False
    def _generate_small_talk_response(self, question: str) -> str:
        """
        Selects a deterministic, professional response based on question type.
        """
        question_lower = question.lower().strip()
        
        # Check what kind of small talk it is (based on the same logic as _is_small_talk)
        if any(phrase in question_lower for phrase in ["hello", "hi", "hey", "good morning", "greetings"]):
            key = "greeting"
        elif any(phrase in question_lower for phrase in ["thank you", "thanks", "i appreciate it"]):
            key = "thanks"
        elif any(phrase in question_lower for phrase in ["who are you", "what is your name"]):
            key = "who_are_you"
        else:
            # Default for other simple small talk (like "how are you")
            key = "generic_positive" 

        # Select a random response from the category
        return random.choice(self.SMALL_TALK_RESPONSES.get(key, self.SMALL_TALK_RESPONSES['greeting']))
    def _tokenize_search_hits(self, cand, docs):
        tok = cand.lower()
        hits = []
        for d in docs:
            if tok in d.get("text","").lower():
                hits.append(d)
        return hits
    
    def deduplicate_docs(self, retrieved_docs):
        seen_texts = set()
        unique_docs = []
        for d in retrieved_docs:
            text_norm = d.get("text", "").strip().lower()
            if text_norm not in seen_texts:
                seen_texts.add(text_norm)
                unique_docs.append(d)
        return unique_docs

    @staticmethod
    def fix_name_spacing(text: str) -> str:
        """Fix spacing issues in names and titles that were converted to camelCase."""
        # Common title and name patterns that need spacing
        spacing_fixes = {
            # Executive titles
            r'\bChiefExecutiveOfficer\b': 'Chief Executive Officer',
            r'\bChiefFinancialOfficer\b': 'Chief Financial Officer',
            r'\bChiefOperatingOfficer\b': 'Chief Operating Officer',
            r'\bChiefTechnologyOfficer\b': 'Chief Technology Officer',
           
            # Department names
            r'\bCorporateDepartments\b': 'Corporate Departments',
            r'\bCommercialDepartments\b': 'Commercial Departments',
            r'\bFleetandEnvironmentalStrategy\b': 'Fleet and Environmental Strategy',
            r'\bShipManagement\b': 'Ship Management',
            r'\bHumanCapital\b': 'Human Capital',
            r'\bGlobalHumanResources\b': 'Global Human Resources',
            r'\bEnterpriseTransformation\b': 'Enterprise Transformation',
            r'\bEnterpriseTransformation-EuropeandAmericas\b': 'Enterprise Transformation-Europe and Americas',
           
            # Committee names
            r'\bShipManagementCommittee\b': 'Ship Management Committee',
            r'\bHumanCapitalCommittee\b': 'Human Capital Committee',
            r'\bBudgetCommittee\b': 'Budget Committee',
            r'\bMarineSafetyCommittee\b': 'Marine Safety Committee',
            r'\bFleetStrategyCommittee\b': 'Fleet Strategy Committee',
            r'\bDXCommittee\b': 'DX Committee',
           
            # Business growth regions
            r'\bBusinessGrowth-Asia\b': 'Business Growth-Asia',
            r'\bBusinessGrowth-MiddleEast\b': 'Business Growth-Middle East',
            r'\bBusinessGrowth-Europe/Africa\b': 'Business Growth-Europe/Africa',
            r'\bBusinessGrowth-Americas\b': 'Business Growth-Americas',
           
            # Other common patterns
            r'\bEXECUTIVEOFFICERS\b': 'Executive Officers',
            r'\bGLOBAL/REGIONALDIRECTORS\b': 'Global/Regional Directors',
            r'\bGLOBALDIRECTORS\b': 'Global Directors',
            r'\bREGIONALDIRECTORS\b': 'Regional Directors',
        }
       
        # Apply all spacing fixes
        for pattern, replacement in spacing_fixes.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
       
        return text

    

    def ask(self, user_id, question):
        
        # --- Step 0: Classify the message (Using the efficient _is_small_talk) ---
        is_small_talk = self._is_small_talk(question)
        print("is_small_talk:", is_small_talk)
        
        if is_small_talk:
            # --- Small Talk Handling ---
            answer = self._generate_small_talk_response(question)

            # Update memory
            memory = self.get_memory(user_id)
            memory.chat_memory.add_user_message(question)
            memory.chat_memory.add_ai_message(answer)
            
            return {"answer": answer, "related": False}
        
        # --- Step 1: Retrieve memory and rewrite query if needed ---
        memory = self.get_memory(user_id)
        past_context = memory.load_memory_variables({}).get("history", "")

        # --- Step 2: Build retrieval query and enrich with deterministic hints ---
        if past_context:
            query_for_retrieval = self._rewrite_query(past_context, question)
        else:
            query_for_retrieval = question

        # --- Step 2.5: Enrich query with retrieval hints ---
        query_for_retrieval = self._add_retrieval_hints(query_for_retrieval)
        print("[DEBUG] Enriched query with retrieval hints:", query_for_retrieval)

        # --- Step 3: Vector Retrieval ---
        # --- Step 3: Vector Retrieval (No change in top-level flow) ---
        # Embedding: use the OpenAI embeddings API properly and serialize
        emb = self.llm.embeddings.create(model="text-embedding-3-large", input=query_for_retrieval)
        query_embedding = emb.data[0].embedding

        # initial vector search (based on the rewritten query)
        retrieved_docs = self.search_cosmos_documents(query_embedding)

        # --- Step 3.5: General Policy Section Title Extraction & Fallback ---
        # Extract canonical policy section titles from the question and use them for targeted fallback
        q_lower = question.lower()
        policy_section_keywords = self._extract_policy_section_keywords(question, q_lower)
        
        if policy_section_keywords:
            print(f"[DEBUG] Extracted policy section keywords: {policy_section_keywords}")
            existing_ids = {d.get('id') for d in retrieved_docs}
            for keyword in policy_section_keywords:
                safe_kw = keyword.replace("'", "''")
                fallback_query = f"""
                    SELECT TOP 10 c.id, c.text, c.metadata
                    FROM c
                    WHERE CONTAINS(LOWER(c.text), '{safe_kw.lower()}')
                """
                try:
                    fallback_results = list(self.container.query_items(query=fallback_query, enable_cross_partition_query=True))
                    for d in fallback_results:
                        if d.get("id") not in existing_ids:
                            # Calculate priority score based on relevance
                            text_lower = d.get("text", "").lower()
                            priority_score = 0.85
                            
                            # Higher priority if document contains both the keyword and approval-related terms
                            keyword_in_doc = any(kw.lower() in text_lower for kw in keyword.split())
                            has_approval_info = any(term in text_lower for term in ["authorised approvers", "approval", "deliberation", "review"])
                            
                            if keyword_in_doc and has_approval_info:
                                priority_score = 0.90
                            
                            retrieved_docs.append({
                                "id": d.get("id"), 
                                "text": d.get("text", ""), 
                                "metadata": d.get("metadata", {}), 
                                "score": priority_score
                            })
                            existing_ids.add(d.get("id"))
                    if fallback_results:
                        print(f"[DEBUG] Policy section keyword fallback found {len(fallback_results)} docs for '{keyword}'")
                except Exception as e:
                    print(f"[DEBUG] Policy section keyword fallback error for '{keyword}': {e}")

        # --- Step 4: Dynamic Keyword + Subtype Extraction & Robust Fallback ---
        # Use the LLM to extract the high-value canonical keyword (e.g., "Novation of Time Charter Contract")
        # keyword_to_check = self._extract_keywords(question)  # existing method you have
        # New helper: extract specific subtype/entity (Time Charter, P&I, FDD, IT, etc.)
        subtype = None
        try:
            # Try a quick deterministic entity detection using regex/keywords first to avoid costcandi
            candidate_subtypes = [
                "time charter", "time charterer", "charter", "novation",
                "p&i", "fdd", "cli",
                "it", "it-related", "software", "system", "implementation",
                "service agreement"
            ]
            q_lower = (question + " " + (past_context or "")).lower()
            for cand in candidate_subtypes:
                if cand in q_lower:
                    subtype = cand
                    break
            # If no subtype found heuristically, ask the LLM for a single subtype label
            if not subtype:
                subtype_prompt = (
                    "Extract a single canonical subtype or domain-word from the question if present (e.g. 'Time Charter', 'P&I', 'IT', 'Service agreement'). "
                    "Return 'None' if no subtype is present.\n\nQuestion:\n" + question
                )
                resp = self.llm.chat.completions.create(
                    model="gpt-4.1",
                    messages=[{"role": "user", "content": subtype_prompt}],
                    temperature=0.0
                )
                subtype = resp.choices[0].message.content.strip().split('\n')[0]
                if subtype.lower() in {"none", ""}:
                    subtype = None
                else:
                    subtype = subtype
        except Exception as e:
            print(f"[DEBUG] subtype extraction error: {e}")
            subtype = None

        # Build a list of fallback keyword variations to try (ordered)
        # Build a list of fallback keyword variations to try (ordered)
        fallback_candidates = []
        if subtype:
            fallback_candidates.append(subtype)

        # Always prefer subtype if present
        if subtype:
            fallback_candidates.insert(0, subtype)

        # De-duplicate while preserving order
        seen_fc = set()
        fallback_candidates = [c for c in fallback_candidates if c and not (c in seen_fc or seen_fc.add(c))]

        # If we already did vector search, check if any candidate substring appears in retrieved docs
        joined_texts = " ".join([d.get("text", "").lower() for d in retrieved_docs])
        missing_candidates = [c for c in fallback_candidates if c and c.lower() not in joined_texts]

        # If any fallback candidates are missing, run progressive SQL fallback searches (CONTAINS)
        if missing_candidates:
            print(f"[DEBUG] Vector search did not include these candidates in top docs: {missing_candidates}")
            for cand in missing_candidates:
                # SQL safe-escape single quotes
                safe_kw = cand.replace("'", "''")
                # Use CONTAINS to match substrings (more relaxed than exact literal)
                fallback_query = f"""
                    SELECT TOP 20 c.id, c.text, c.metadata
                    FROM c
                    WHERE CONTAINS(LOWER(c.text), '{safe_kw.lower()}')
                """
                try:
                    fallback_results = list(self.container.query_items(query=fallback_query, enable_cross_partition_query=True))
                except Exception as e:
                    print(f"[DEBUG] fallback SQL for '{cand}' error: {e}")
                    fallback_results = []
                # Add non-duplicate docs to retrieved_docs (preserve earlier docs first)
                existing_ids = {d['id'] for d in retrieved_docs}
                added = 0
                for d in fallback_results:
                    if d.get("id") not in existing_ids:
                        retrieved_docs.append({"id": d.get("id"), "text": d.get("text", ""), "metadata": d.get("metadata", {}), "score": 0.0})
                        existing_ids.add(d.get("id"))
                        added += 1
                print(f"[DEBUG] Fallback search for '{cand}' added {added} docs")

            # If still zero matches for all fallback candidates, try a relaxed substring search on keyword tokens
            if all(len(self._tokenize_search_hits(cand, retrieved_docs)) == 0 for cand in fallback_candidates):
                print("[DEBUG] No fallback docs found with CONTAINS; trying relaxed token search on key tokens.")
                tokens = []
                for cand in fallback_candidates:
                    tokens += [t for t in cand.split() if len(t) > 2]
                tokens = list(dict.fromkeys(tokens))[:6]  # unique tokens, limit to 6
                for tok in tokens:
                    safe_tok = tok.replace("'", "''").lower()
                    token_query = f"""
                        SELECT TOP 10 c.id, c.text, c.metadata
                        FROM c
                        WHERE CONTAINS(LOWER(c.text), '{safe_tok}')
                    """
                    try:
                        token_res = list(self.container.query_items(query=token_query, enable_cross_partition_query=True))
                    except Exception as e:
                        token_res = []
                    existing_ids = {d['id'] for d in retrieved_docs}
                    for d in token_res:
                        if d.get("id") not in existing_ids:
                            retrieved_docs.append({"id": d.get("id"), "text": d.get("text", ""), "metadata": d.get("metadata", {}), "score": 0.0})
                            existing_ids.add(d.get("id"))


        retrieved_docs = self.deduplicate_docs(retrieved_docs)
        # --- Focus context on the correct policy section (e.g., IT-related assets) ---
        must_header = (
            "Acquisition, disposal of IT related assets "
            "a) Equipment and fixtures b) Intellectual property "
            "c) Software (including development cost) d) Other IT assets"
        )
        # if keyword_to_check == must_header:
        #     retrieved_docs = self._filter_by_policy_header(retrieved_docs, must_header)

        # --- Step 4.6: Route "service agreement with subsidiaries" questions to the right section ---
        q_lower = question.lower()
        subs_keys = ["mctwtn","mcttky","mctdbi","mctldn","mctrmd","mcthou","mctcph","mctbog","mctrmd","unix","tms"]

        def _is_sa_with_subs(txt: str) -> bool:
            t = txt.lower()
            # strong signals for the correct table - be more flexible with matching
            return (
                "service agreement" in t and (
                    "mctspr subsidiaries" in t or
                    "mctspr" in t or  # More flexible
                    "conclusion / termination / revision of service agreement" in t or
                    "conclusion/ termination/ revision of service agreement" in t or
                    "conclusion termination revision" in t.replace("/", " ").replace("  ", " ") or  # Handle various spacing
                    "responsible department for conclusion / termination / revision of service agreement" in t or
                    "responsible department for conclusion/ termination/ revision of service agreement" in t or
                    "responsible department" in t and "conclusion" in t and "service agreement" in t  # More flexible pattern
                )
            )

        if "service agreement" in q_lower and any(s in q_lower for s in subs_keys):
            # keep only chunks about service agreements with MCTSPR subsidiaries
            filtered = [d for d in retrieved_docs if _is_sa_with_subs(d.get("text",""))]
            
            # If filtering found documents, use them; otherwise keep all retrieved docs but prioritize service agreement ones
            if filtered:
                # optional: trim each chunk so it starts at the correct header
                hdr_pattern = r"(?:^|\n).*service agreement.*mctspr subsidiaries.*"
                for d in filtered:
                    txt = d.get("text","")
                    m = re.search(hdr_pattern, txt, flags=re.IGNORECASE)
                    if m:
                        d["text"] = txt[m.start():]
                    # Boost priority for documents that match both service agreement AND amount context
                    d["priority"] = 3  # base priority for service agreement with subsidiaries
                    # If amount is mentioned in question, boost docs that mention similar amounts
                    amount_match = re.search(r"(?:US\$|USD|\$)\s*([0-9]{1,3}(?:[, ]?[0-9]{3})*|[0-9]+)", question, flags=re.IGNORECASE)
                    if amount_match:
                        # Check if document mentions approval/departments (main query focus)
                        txt_lower = txt.lower()
                        if any(term in txt_lower for term in ["approval", "approver", "department", "responsible", "authorised"]):
                            d["priority"] = 5  # Highest priority: service agreement + approval/departments + amount context
                # Sort by priority (highest first), then by score, then by ID for deterministic ordering
                retrieved_docs = sorted(filtered, key=lambda x: (-x.get("priority", 0), -x.get("score", 0.0), x.get("id", "")))
            else:
                # otherwise, hard-deprioritize generic "contract with MOL and its subsidiaries"
                for d in retrieved_docs:
                    t = d.get("text","").lower()
                    d["priority"] = -1 if ("contract with mol" in t and "subsidiaries" in t) else 0
                # Sort by priority, then by score, then by ID for deterministic ordering
                retrieved_docs = sorted(retrieved_docs, key=lambda x: (-x.get("priority", 0), -x.get("score", 0.0), x.get("id", "")))

        # --- Prefer correct Charter direction and duration window if mentioned ---
        is_charter_in_q = re.search(r"\bcharter\s*in\b", q_lower) is not None
        is_charter_out_q = re.search(r"\bcharter\s*out\b", q_lower) is not None
        dur_match_q = re.search(r"\b(\d+)\s*[- ]?\s*year", q_lower)
        target_years = int(dur_match_q.group(1)) if dur_match_q else None

        def _matches_duration_window(txt: str, years: int, charter_in: bool, charter_out: bool) -> bool:
            t = txt.lower()
            if years is None:
                return False
            if charter_in:
                if years >= 5 and "5 years and more" in t:
                    return True
                if years > 1 and years < 5 and "more than 1 year and less than 5 years" in t:
                    return True
                if years <= 1 and "1 year and less" in t:
                    return True
            if charter_out:
                if years >= 3 and "3 years and more" in t:
                    return True
                if years > 1 and years < 3 and "more than 1 year and less than 3 years" in t:
                    return True
                if years <= 1 and "1 year and less" in t:
                    return True
            return False

        if (is_charter_in_q or is_charter_out_q):
            for d in retrieved_docs:
                txt = d.get("text", "")
                t = txt.lower()
                # Boost correct direction; demote opposite to reduce mixing
                if is_charter_in_q:
                    if "charter out" in t:
                        d["priority"] = d.get("priority", 0) - 2
                    if "charter in" in t:
                        d["priority"] = d.get("priority", 0) + 2
                if is_charter_out_q:
                    if "charter in" in t:
                        d["priority"] = d.get("priority", 0) - 2
                    if "charter out" in t:
                        d["priority"] = d.get("priority", 0) + 2
                # Boost exact duration window if identifiable
                if target_years is not None and _matches_duration_window(txt, target_years, is_charter_in_q, is_charter_out_q):
                    d["priority"] = d.get("priority", 0) + 3
            # Re-sort after boosting
            retrieved_docs = sorted(retrieved_docs, key=lambda x: (-x.get("priority", 0), -x.get("score", 0.0), x.get("id", "")))


        # --- Step 5: Preparing Context (Improved debug block) ---
        # Ensure deterministic document ordering: sort by priority (if set), then score (desc), then by ID (asc) for stability
        # This ensures consistent document order across multiple runs
        retrieved_docs = sorted(retrieved_docs, key=lambda x: (
            -x.get("priority", 0),  # Priority first (if set, higher is better)
            -x.get("score", 0.0),   # Then by score (higher is better)
            x.get("id", "")         # Finally by ID for deterministic tie-breaking
        ))
        
        context = self.format_context(retrieved_docs)
        print(f"[DEBUG] Retrieved docs count (post-fallback merge): {len(retrieved_docs)}")



        # Print top few retrieved document snippets for verification
        for i, d in enumerate(retrieved_docs[:6]):
            preview = d.get("text", "").replace("\n", " ").strip()[:180]
            print(f"[DEBUG] Doc {i+1} | Score: {d.get('score', 0):.3f} | Snippet: {preview}...")

        # Save retrieved sections to debug file
        self._save_retrieved_sections(question, query_for_retrieval, retrieved_docs, subtype, fallback_candidates)

        # Save full retrieval trace for external inspection
        with open("Verification_retrieved_docs.txt", "a", encoding="utf-8") as f:
            f.write(f"\n\n==============================\n")
            f.write(f"Timestamp: {datetime.utcnow().isoformat()}\n")
            f.write(f"Question: {question}\n")
            f.write(f"Rewritten Query: {query_for_retrieval}\n")
            # f.write(f"Extracted Keyword: {keyword_to_check}\n")
            f.write(f"Extracted Subtype: {subtype}\n")
            f.write(f"Final Fallback Candidates: {fallback_candidates}\n")
            f.write(f"Retrieved Docs: {len(retrieved_docs)}\n\n")
            for i, d in enumerate(retrieved_docs):
                f.write(f"Doc {i+1} (Score {d.get('score', 0):.3f}):\n{d.get('text','')}\n\n")
            f.write("==============================\n")
        # --- Step 4: Generate Final Answer ---
        # Build conditional extra rules based on question content
        extra_rules = ""
        if any(x in question.lower() for x in ["novation", "amendment", "cancellation"]):
            extra_rules += self.NOVATION_RULES
        if any(x in question.lower() for x in ["cost", "fee", "amount", "budget", "it-related"]):
            extra_rules += self.COST_RULES
        if any(x in question.lower() for x in ["cli", "fdd", "dth", "tcl", "subtype"]):
            extra_rules += self.SUBTYPE_RULES
        if any(x in question.lower() for x in ["committee", "member", "chairperson", "secretariat", "head of department"]):
            extra_rules += self.COMMITTEE_RULES
        # Add duration selection rule when question mentions years/tenor
        if re.search(r"\b\d+\s*[- ]?\s*year", question, flags=re.IGNORECASE):
            DURATION_RULES = """
            If the question involves a period/tenor (e.g., N years) for Time Charter:
            - Identify all duration thresholds in the Document Context (e.g., "5 years and more", "More than 1 year and less than 5 years", "1 year and less", "3 years and more").
            - ALWAYS select the single threshold that exactly covers the given period.
            - For example, "Charter in for 3 years" MUST use "More than 1 year and less than 5 years".
            - Do not mix rows across different duration windows or directions (Charter in vs Charter out).
            """
            extra_rules += DURATION_RULES

        # --- Step 4: Generate Final Answer ---
        # The final prompt includes everything: the original question, memory, and context.
        # We now trust the memory + LLM to use the best information.
        # You must **only extract lines directly related to the user’s topic** (for example, if the user asks about *product acquisition*, do not include other categories like *claim settlement* or *consultancy agreements*). 
# Do not summarize or interpret — copy the exact relevant block only.

        prompt = f"""
You are **Thinkpalm's Corporate Knowledge Assistant**. 
Your job is to produce an **exact, policy-faithful answer** using *only* the information from the Document Context below.

⚠️ STRICT RULES
1. **PRIORITIZE THE MAIN QUESTION FOCUS**: Always answer the main question first (e.g., "approval and departments", "who approves", "what is the process") before mentioning amount thresholds.
   - If the question asks "What approval and departments are involved for X with amount Y?", focus on answering "approval and departments for X" first
   - Then include the specific amount threshold (e.g., USD800,000) as supporting context
   - Do NOT let generic amount thresholds (e.g., "more than 50000") override the main query focus
2. Answer **only** from the Document Context.  
   - If not enough info exists, reply exactly:  
     "I do not have sufficient information in the available policy context to answer that."
   - Never generalize, assume, or infer missing policy details.
3. Use **verbatim wording** for entities such as "Authorised Approver", "Co-Management Dept.", "Deliberation", "Review", "CC", "Chairperson", "Head of Department", and "Secretariat".
4. Do **not** summarize, rename, or interpret policies — copy exact table lines that apply.
5. For committee-related questions: Include ALL structural information (Chairperson, Members, Head of Department/Secretariat) even if the question only asks about one aspect.
6. Merge related clauses logically if they describe the same policy action (e.g., "Novation", "Amendment").
7. **CRITICAL - Amount Threshold Selection**: When multiple amount thresholds apply to the same amount, ALWAYS use the MOST SPECIFIC threshold.
   - For example: $8,300 qualifies for both "Less than US$50,000" and "Less than US$25,000"
   - You MUST use "Less than US$25,000" (the more specific), NOT "Less than US$50,000" (the general)
   - Always identify ALL applicable thresholds, then select the MOST SPECIFIC one
   - When multiple departments or thresholds appear, clearly state which rule applies and under what condition
8. For service agreement questions with MCTSPR subsidiaries: Prioritize documents that specifically mention "Conclusion / Termination / Revision of service agreement with MCTSPR subsidiaries" over generic contract documents.
9. Maintain a concise professional style:
   - **Opening Summary:** one line answering the main question directly.  
   - **Details:** bulleted or numbered list of facts from the table (approval, departments, etc.).  
   - **Amount Context:** if a specific amount was mentioned, include the relevant threshold/rule that applies to that amount.
   - **Conclusion:** short sentence summarizing the rule or required action.

{extra_rules}

INPUTS
Conversation History:
{past_context}

Document Context:
{context}

User Question:
{question}

OUTPUT
Answer:
"""



        
        response = self.llm.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0  # Set to 0 for deterministic answers
        )
        answer = response.choices[0].message.content.strip()
        
        # Fix spacing issues in names and titles
        answer = self.fix_name_spacing(answer)
        
        # --- Step 5: Update Memory and Cache ---
        memory.chat_memory.add_user_message(question)
        memory.chat_memory.add_ai_message(answer)
        self.save_chat_message(user_id, question, answer)
        # self.last_question_cache[user_id] = question # <<< Remove this line, as it's no longer needed
        with open("Verification_retrieved_docs.txt", "a", encoding="utf-8") as f:
            f.write(f"\n\nAnswer\n==========================\n{answer}\n\n")
        return {"answer": answer, "related": bool(past_context)} # 'related' is now simply based on whether a history exists
        

# ------------------------------------------------------------
# Microsoft Teams Bot Integration
# ------------------------------------------------------------
class ThinkpalmRAGBot(ActivityHandler):
    def __init__(self):
        COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
        COSMOS_KEY = os.getenv("COSMOS_KEY")
        COSMOS_DATABASE = os.getenv("COSMOS_DATABASE")
        COSMOS_CONTAINER = os.getenv("COSMOS_CONTAINER")
        CHAT_CONTAINER = os.getenv("CHAT_CONTAINER")

        self.rag = ThinkpalmCosmosRAGmethod2(
            COSMOS_ENDPOINT,
            COSMOS_KEY,
            COSMOS_DATABASE,
            COSMOS_CONTAINER,
            CHAT_CONTAINER
        )

    async def on_members_added_activity(self, members_added: list[ChannelAccount], turn_context: TurnContext):
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                await turn_context.send_activity("Hello and welcome to Thinkpalm's Knowledge Assistant!")

    async def on_message_activity(self, turn_context: TurnContext):
        user_id = turn_context.activity.from_property.id or "default_user"
        user_msg = str(turn_context.activity.text).strip()

        # Send initial typing indicator
        typing_activity = Activity(type=ActivityTypes.typing)
        await turn_context.send_activity(typing_activity)

        # Create a background task to send periodic typing indicators
        import asyncio
        
        async def send_typing_periodically():
            """Send typing indicator every 3 seconds while processing"""
            try:
                while True:
                    await asyncio.sleep(3)  # Wait 3 seconds
                    await turn_context.send_activity(typing_activity)
            except asyncio.CancelledError:
                pass  # Task was cancelled, which is expected
        
        # Start the typing indicator task
        typing_task = asyncio.create_task(send_typing_periodically())
        
        try:
            # Run blocking RAG call in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.rag.ask, user_id, user_msg)
            answer = response["answer"]
        finally:
            # Cancel the typing indicator task when done
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass
        
        # Send the response
        await turn_context.send_activity(MessageFactory.text(answer))


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
# """
if __name__ == "__main__":
    # COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
    # COSMOS_KEY = os.getenv("COSMOS_KEY")
    # DB_NAME = "thinkpalm_db"
    # DOCS_CONTAINER = "ragEmbedding"
    
    COSMOS_ENDPOINT=os.getenv("COSMOS_ENDPOINT")

    COSMOS_KEY=os.getenv("COSMOS_KEY")

    COSMOS_DATABASE=os.getenv("COSMOS_DATABASE")

    COSMOS_CONTAINER=os.getenv("COSMOS_CONTAINER")
        
        
    CHAT_CONTAINER = os.getenv("CHAT_CONTAINER")
    # print(COSMOS_ENDPOINT, COSMOS_KEY, COSMOS_DATABASE, COSMOS_CONTAINER, CHAT_CONTAINER)
    # exit()
    rag = ThinkpalmCosmosRAGmethod2(COSMOS_ENDPOINT, COSMOS_KEY, COSMOS_DATABASE, COSMOS_CONTAINER, CHAT_CONTAINER)

    # answer = rag.ask("user_id", "hi")
    # print(f"Assistant: {answer}\n")
    # user_id = input("Enter user_id: ")
    user_id= "test_user"

    print("\nType 'exit' to stop chatting.\n")
    while True:
        user_msg = input("You: ")
        if user_msg.lower() in ["exit", "quit"]:
            print("Ending chat...")
            break
        response = rag.ask(user_id, user_msg)
        ans, related = response["answer"], response["related"]
        print(f"Assistant: {ans}\n, {related}\n")
        # 
# """
