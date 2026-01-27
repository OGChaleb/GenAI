#Start: uvicorn app:app --host 0.0.0.0 --port 8000
#Venv: .\venv\Scripts\Activate.ps1

# =========================
# app.py – ProfessorGPT
# AutoGen + Haystack + FastAPI
# =========================

import logging
import asyncio
import json
import re
import time
import hashlib
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader

# Haystack
from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders import PromptBuilder

# AutoGen
from autogen import AssistantAgent, UserProxyAgent

# TruLens (optional)
try:
    from trulens_eval.feedback.provider import LiteLLM as LiteLLMProvider
except Exception:  # noqa: BLE001
    LiteLLMProvider = None

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------------------------------
# FastAPI App
# -------------------------------------------------
app = FastAPI(title="ProfessorGPT – Email RAG Agent System")

# Simple in-memory dedup cache to avoid double-processing identical payloads
DEDUP_TTL_SECONDS = 90
REQUEST_CACHE: Dict[str, Dict[str, Any]] = {}
LOG_EXPORT_DIR = Path(__file__).parent / "logs"

# -------------------------------------------------
# PDF INGESTION
# -------------------------------------------------
PDF_FOLDER = r"D:\GenAI\Haystack2\Haystack RAG"


def export_request_logs(log_text: str) -> Path:
    """Persist captured logs for a single request to a timestamped file."""
    LOG_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    file_path = LOG_EXPORT_DIR / f"request-log-{timestamp}.txt"
    file_path.write_text(log_text, encoding="utf-8")
    return file_path

def split_text(text: str, size=600, overlap=120) -> List[str]:
    chunks = []
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


logger.info("Loading PDFs...")
documents: List[Document] = []

for pdf_path in Path(PDF_FOLDER).glob("*.pdf"):
    reader = PdfReader(pdf_path)
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text:
            continue
        for i, chunk in enumerate(split_text(text), start=1):
            documents.append(
                Document(
                    content=chunk,
                    meta={
                        "source": pdf_path.name,
                        "page": page_num,
                        "chunk": i
                    }
                )
            )

logger.info(f"Loaded {len(documents)} chunks")

# -------------------------------------------------
# Haystack Components
# -------------------------------------------------
document_store = InMemoryDocumentStore()
document_store.write_documents(documents)

retriever = InMemoryBM25Retriever(
    document_store=document_store,
    top_k=3
)

 
PROMPT_TEMPLATE = """
You are a precise academic assistant.
Answer the question using ONLY the information from the context.
If the answer is not present, reply exactly with: I don't know.

Context:
{% for document in documents %}
[{{ document.meta.source }} | p.{{ document.meta.page }} | c{{ document.meta.chunk }}]
{{ document.content }}

{% endfor %}

Question: {{ question }}
Answer:
"""

prompt_builder = PromptBuilder(
    template=PROMPT_TEMPLATE,
    required_variables={"documents", "question"}
)

# -------------------------------------------------
# Ollama LLM Wrapper
# -------------------------------------------------
class OllamaLLM:
    def __init__(self, model="gpt-oss:20b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def run(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.0
            },
            timeout=300
        )
        return response.json().get("response", "").strip()


llm = OllamaLLM()

# -------------------------------------------------
# Helper to extract JSON from agent responses
# -------------------------------------------------
def extract_json_from_response(response: str, is_array: bool = False):
    """Extract JSON object or array from agent response text."""
    response = response.replace("TERMINATE", "").strip()
    
    # Try to find JSON object or array
    start_char = "[" if is_array else "{"
    start_idx = response.rfind(start_char)  # Use rfind to get the last occurrence
    
    if start_idx == -1:
        raise ValueError(f"No JSON {'array' if is_array else 'object'} found in response: {response[:200]}")
    
    # Try to find the matching closing bracket
    response_from_start = response[start_idx:]
    if is_array:
        # Find the matching ]
        bracket_count = 0
        for i, char in enumerate(response_from_start):
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    return json.loads(response_from_start[:i+1])
    else:
        # Find the matching }
        bracket_count = 0
        for i, char in enumerate(response_from_start):
            if char == '{':
                bracket_count += 1
            elif char == '}':
                bracket_count -= 1
                if bracket_count == 0:
                    return json.loads(response_from_start[:i+1])
    
    # Fallback: try to parse the whole substring
    return json.loads(response_from_start)

def extract_json_from_chat_history(chat_history: list, agent_name: str, is_array: bool = False):
    """Extract JSON from first valid agent response in chat history."""
    # Try to find the first message that contains valid JSON
    for message in chat_history:
        if isinstance(message, dict) and "content" in message:
            content = message.get("content", "")
            # Skip empty or very short messages
            if not content or len(content.strip()) < 5:
                continue
            
            try:
                return extract_json_from_response(content, is_array)
            except (ValueError, json.JSONDecodeError):
                # This message doesn't contain valid JSON, try the next one
                continue
    
    raise ValueError(f"No valid JSON found in chat history from agent {agent_name}")

def is_terminated(chat_result, accept_json: bool = False) -> bool:
    for msg in chat_result.chat_history:
        content = msg.get("content", "")
        if "TERMINATE" in content:
            return True

        if accept_json:
            try:
                json.loads(content)
                return True
            except Exception:
                pass
    return False


def haystack_query(question: str) -> dict:
    """Retrieve documents and generate an answer using Haystack.
    Returns both the raw documents and the LLM answer."""
    logger.info(f"Haystack query: {question}")
    retrieved = retriever.run(query=question)
    docs = retrieved["documents"]
    
    logger.info(f"Retrieved {len(docs)} documents")
    for i, doc in enumerate(docs, 1):
        logger.info(f"  Doc {i}: {doc.meta.get('source', 'unknown')} p.{doc.meta.get('page', '?')} - {doc.content[:100]}...")

    prompt = prompt_builder.run(
        documents=docs,
        question=question
    )["prompt"]

    answer = llm.run(prompt)
    logger.info(f"Haystack answer: {answer[:200]}...")
    
    return {
        "question": question,
        "documents": docs,
        "answer": answer
    }

# -------------------------------------------------
# AutoGen Configuration
# -------------------------------------------------
OLLAMA_CONFIG = {
    "config_list": [
        {
            "model": "gpt-oss:20b",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama"
        }
    ],
    "temperature": 0.0
}

trulens_provider = None
if LiteLLMProvider is not None:
    try:
        trulens_provider = LiteLLMProvider(
            model_engine="ollama/gpt-oss:20b",
            api_base="http://localhost:11434/v1",
            api_key="ollama"
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"TruLens LiteLLM provider init failed, continuing without TruLens: {exc}")

user_proxy = UserProxyAgent(
    name="Orchestrator",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,
    is_termination_msg=lambda x: "TERMINATE" in x.get("content", "")
)

intent_agent = AssistantAgent(
    name="IntentAnalyst",
    llm_config=OLLAMA_CONFIG,
    system_message="""
You are a meticulous academic email analyst. 
Your job is to determine the purpose (intent) of the email, detect if personal data is present, and identify all specific information that would be required to respond accurately.

Tasks:
1. Identify the intent of the email (e.g., "request for information", "submission of documents", "question about syllabus", "administrative request", "none").
2. Extract a list of all required pieces of information needed to respond to the email.
3. Identify any personal data in the email (names, emails, dates, IDs).

Constraints:
- Use only the content of the email.
- Do not infer anything not explicitly stated.
- Return valid JSON **only**, exactly in this format:

{
  "intent": "...",
  "required_information": ["...", "..."],
  "personal_data_detected": ["...", "..."]
}
TERMINATE

"""
)

question_agent = AssistantAgent(
    name="QuestionEngineer",
    llm_config=OLLAMA_CONFIG,
    system_message="""
You are a precise academic question engineer.
Based on the information needs extracted from an email, generate clear factual questions suitable for retrieving knowledge from academic documents.

Tasks:
1. Transform each required information item into a factual, answerable question.
2. Keep each question concise and specific.
3. Avoid questions that cannot be answered from academic materials.
4. Return valid JSON **only**, as an array of objects, each with a "question" key:

[
  {"question": "..."},
  {"question": "..."}
]
TERMINATE

"""
)

composer_agent = AssistantAgent(
    name="ResponseComposer",
    llm_config=OLLAMA_CONFIG,
    max_consecutive_auto_reply=1,
    system_message="""
You are an academic assistant tasked with composing a professional reply email in the name of Prof. Dr. Manuel Fritz, from the Business School Pforzheim.
Use ONLY the documents and Q&A pairs provided. 

Tasks:
1. Compose a polite, formal, and precise email reply.
2. Answer all questions using ONLY the provided knowledge.
3. If the answer is not in the source documents, explicitly say "I don't know."
4. Do NOT invent facts, opinions, or data.
5. Maintain academic tone, clarity, and proper grammar.
6. Ensure the response is structured like a professional email.
7. ONLY when the context really implies that the sender could need appointment, refer to this Link where appointments can be booked: https://calendly.com/manuelfritz/meeting
8. Do not use "**" to highlight text. Since the final output will directly go into an email, avoid any markdown formatting.

Input:
- Original email text
- Source documents (PDF excerpts)
- Q&A pairs from the retrieval step

Output:
- Plain text of the reply email
- Append TERMINATE at the end
IMPORTANT: Do NOT include citations, page numbers, or source references in the final email. Paraphrase all content from the documents in your own words.

"""
)

privacy_agent = AssistantAgent(
    name="PrivacyAgent",
    llm_config=OLLAMA_CONFIG,
    max_consecutive_auto_reply=1,
    system_message="""
You are a privacy and GDPR compliance expert.
Your task is to review an email draft and ensure that:

1. No unnecessary personal data (names, emails, IDs, dates) is included, except the professors information within the signature and the Sender's Name when greeting.
2. Sensitive information is anonymized or removed.
3. The message remains professional, coherent, and grammatically correct.
4. The content does not inadvertently disclose private or sensitive data.

Output:
- Return ONLY the final, privacy-compliant email text.
- Do NOT include commentary.
- Append TERMINATE at the end.

"""
)

# -------------------------------------------------
# AutoGen Workflow
# -------------------------------------------------
def run_email_pipeline(email_text: str) -> Dict[str, Any]:
    # 1. Intent analysis
    logger.info("=== Step 1: Intent Analysis ===")
    intent_result = user_proxy.initiate_chat(
    intent_agent,
    message=email_text,
    clear_history=True
)

    if not is_terminated(intent_result, accept_json=True):
        raise RuntimeError("IntentAnalyst did not terminate or return valid JSON")


    try:
        intent_data = extract_json_from_chat_history(
            intent_result.chat_history,
            "IntentAnalyst",
            is_array=False
        )
    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"Failed to parse intent JSON: {e}")
        raise


    logger.info(f"Intent analysis: {intent_data}")

    # 2. Question generation
    logger.info("=== Step 2: Question Generation ===")
    questions = []

    # Only generate questions if there is required information
    required_info = intent_data.get("required_information") or []
    if len(required_info) > 0:
        questions_result = user_proxy.initiate_chat(
            question_agent,
            message=json.dumps(required_info),
            clear_history=True
        )

        if not is_terminated(questions_result, accept_json=True):
            raise RuntimeError("QuestionEngineer did not terminate or return valid JSON")

        try:
            questions = extract_json_from_chat_history(
                questions_result.chat_history,
                "QuestionEngineer",
                is_array=True
            )
        except (ValueError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse questions JSON: {e}")
            logger.warning("Falling back to empty questions list")
            questions = []
    else:
        logger.info("No required information identified, skipping question generation")

    logger.info(f"Generated {len(questions)} questions")

    # 3. Haystack retrieval
    logger.info("=== Step 3: Haystack Retrieval ===")
    all_retrieved_docs = []
    qa_pairs = []
    
    for q in questions:
        question_text = q.get("question", q) if isinstance(q, dict) else q
        result = haystack_query(question_text)
        qa_pairs.append({
            "question": result["question"],
            "answer": result["answer"]
        })
        all_retrieved_docs.extend(result["documents"])
    
    # Remove duplicates from retrieved docs while preserving order
    seen_content = set()
    unique_docs = []
    for doc in all_retrieved_docs:
        if doc.content not in seen_content:
            seen_content.add(doc.content)
            unique_docs.append(doc)
    
    # Format documents for composer
    formatted_docs = ""
    contexts = []
    for i, doc in enumerate(unique_docs, 1):
        contexts.append(
            f"[{doc.meta.get('source', 'unknown')} p.{doc.meta.get('page', '?')} c{doc.meta.get('chunk', '?')}] {doc.content}"
        )
        formatted_docs += f"""\n[Document {i}: {doc.meta.get('source', 'unknown')} | p.{doc.meta.get('page', '?')}]
{doc.content}
{'-'*80}"""
    
    # Also include Q&A pairs for reference
    qa_section = ""
    for qa in qa_pairs:
        qa_section += f"\n\nQ: {qa['question']}\nA: {qa['answer']}"
    
    logger.info(f"Retrieved {len(unique_docs)} unique documents for composer")
    
    # 4. Draft email
    logger.info("=== Step 4: Compose Draft ===")
    draft_result = user_proxy.initiate_chat(
        composer_agent,
        message=f"""
Original Email:
{email_text}

SOURCE DOCUMENTS (from syllabus - use these as primary reference):
{formatted_docs}

QUESTION-ANSWER PAIRS (for context):
{qa_section}

Draft a professional reply using ONLY the information from the SOURCE DOCUMENTS above.
""",
        clear_history=True
    )
    if not is_terminated(draft_result):
        raise RuntimeError("ResponseComposer did not TERMINATE")

    draft = draft_result.chat_history[-1]["content"]
    draft = draft.replace("TERMINATE", "").strip()
    logger.info(f"Draft email: {draft[:200]}...")

    # 5. Privacy check
    logger.info("=== Step 5: Privacy Review ===")
    privacy_result = user_proxy.initiate_chat(
    privacy_agent,
    message=draft,
    clear_history=True
)

    if not is_terminated(privacy_result):
        raise RuntimeError("PrivacyAgent did not TERMINATE")

    final_email = privacy_result.chat_history[-1]["content"]
    final_email = final_email.replace("TERMINATE", "").strip()

    logger.info(f"Final email: {final_email}")

    return {
        "final_email": final_email,
        "qa_pairs": qa_pairs,
        "contexts": contexts,
        "intent": intent_data,
        "questions": [q.get("question", q) if isinstance(q, dict) else q for q in questions],
    }


def process_email_with_agents(email_text: str) -> str:
    result = run_email_pipeline(email_text)
    return result["final_email"]


def evaluate_email_with_trulens(email_text: str) -> Dict[str, Any]:
    pipeline_result = run_email_pipeline(email_text)

    transparency_score: Optional[float] = None
    transparency_reason: Optional[str] = None
    completeness_score: Optional[float] = None
    completeness_reason: Optional[str] = None

    # Fallback heuristics when TruLens provider is unavailable.
    # Transparency: token overlap between final email and retrieved contexts.
    email_tokens = set(re.findall(r"\w+", pipeline_result["final_email"].lower()))
    context_tokens = set(re.findall(r"\w+", " ".join(pipeline_result["contexts"]).lower()))
    if email_tokens:
        overlap = email_tokens & context_tokens
        transparency_score = round(len(overlap) / len(email_tokens), 3)
        transparency_reason = "Fraction of email tokens present in retrieved context (heuristic)."

    # Completeness: fraction of questions not answered with "I don't know".
    answered = 0
    for qa in pipeline_result["qa_pairs"]:
        answer_text = qa.get("answer", "").strip().lower()
        if answer_text and "i don't know" not in answer_text:
            answered += 1
    if pipeline_result["qa_pairs"]:
        completeness_score = round(answered / len(pipeline_result["qa_pairs"]), 3)
        completeness_reason = "Fraction of generated questions answered without 'I don't know' (heuristic)."

    # Log TruLens evaluation results
    logger.info("=== TruLens Evaluation ===")
    logger.info(f"Transparency Score: {transparency_score} - {transparency_reason}")
    logger.info(f"Completeness Score: {completeness_score} - {completeness_reason}")

    return pipeline_result

# -------------------------------------------------
# API Models
# -------------------------------------------------
class EmailRequest(BaseModel):
    email_subject: str = ""
    email_body: str = ""
    email_sender: str = ""
    email_date: str = ""
    email_attachment: str = ""
    evaluate_with_trulens: bool = False

class EmailResponse(BaseModel):
    subject: str
    body: str
    sender_email: str

# -------------------------------------------------
# API Endpoint (for n8n)
# -------------------------------------------------
@app.post("/process-email", response_model=EmailResponse)
async def process_email(payload: EmailRequest):
    log_buffer = io.StringIO()
    buffer_handler = logging.StreamHandler(log_buffer)
    buffer_handler.setLevel(logging.INFO)
    buffer_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(buffer_handler)
    log_file_path: Optional[Path] = None

    try:
        payload_dict = payload.model_dump()
        logger.info(f"Received payload: {payload_dict}")

        # Deduplicate identical requests within a short window to avoid double drafts
        fingerprint = hashlib.sha256(
            json.dumps(payload_dict, sort_keys=True).encode("utf-8")
        ).hexdigest()
        now = time.time()

        # Prune expired cache entries
        expired_keys = [k for k, v in REQUEST_CACHE.items() if now - v["ts"] > DEDUP_TTL_SECONDS]
        for k in expired_keys:
            REQUEST_CACHE.pop(k, None)

        cached = REQUEST_CACHE.get(fingerprint)
        if cached and now - cached["ts"] <= DEDUP_TTL_SECONDS:
            logger.info("Duplicate payload detected; returning cached response")
            return cached["response"]
        
        full_email_text = f"""
    Subject: {payload.email_subject}
    From: {payload.email_sender}
    Date: {payload.email_date}

    Email Body:
    {payload.email_body}

    ----------------------------------------
    Attachments / Additional Content:
    {payload.email_attachment}
    """.strip()

        if payload.evaluate_with_trulens:
            result = await asyncio.to_thread(
                evaluate_email_with_trulens,
                full_email_text
            )
        else:
            result = await asyncio.to_thread(
                run_email_pipeline,
                full_email_text
            )

        # Parse final email to extract subject and body
        final_email = result["final_email"]
        lines = final_email.split("\n")
        
        subject = ""
        body_start_idx = 0
        
        # Look for Betreff: or Subject: line
        for i, line in enumerate(lines):
            lower_line = line.lower().strip()
            if lower_line.startswith("betreff:"):
                subject = line[line.lower().index("betreff:") + 8:].strip()
                body_start_idx = i + 1
                break
            elif lower_line.startswith("subject:"):
                subject = line[line.lower().index("subject:") + 8:].strip()
                body_start_idx = i + 1
                break
        
        # If no subject line found, use first non-empty line
        if not subject:
            for i, line in enumerate(lines):
                if line.strip():
                    subject = line.strip()
                    body_start_idx = i + 1
                    break
        
        # Everything after subject line is body, skip empty lines at start
        body_lines = lines[body_start_idx:]
        while body_lines and not body_lines[0].strip():
            body_lines.pop(0)
        body = "\n".join(body_lines).strip()
        
        # Extract sender email from "Name <email@example.com>" format
        sender_email = payload.email_sender
        if "<" in sender_email and ">" in sender_email:
            sender_email = sender_email[sender_email.index("<") + 1:sender_email.index(">")]
        
        response_payload = {"subject": subject, "body": body, "sender_email": sender_email}
        REQUEST_CACHE[fingerprint] = {"ts": now, "response": response_payload}

        log_file_path = export_request_logs(log_buffer.getvalue())
        logger.info(f"Request log exported to {log_file_path}")

        return response_payload

    except Exception as e:
        logger.exception("Processing failed")
        if log_buffer.tell() > 0:
            log_file_path = export_request_logs(log_buffer.getvalue())
            logger.error(f"Request log exported to {log_file_path} after error")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        logger.removeHandler(buffer_handler)
        log_buffer.close()

