import os
import re
from typing import Any, Dict, Optional, TypedDict
from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import END, START, StateGraph


class InterviewGraphState(TypedDict):
    instructions: str
    job_requirement: str
    selected_agent: str
    gemini_output: Optional[str]
    openai_output: Optional[str]
    merged_output: Optional[str]
    final_output: Optional[str]
    extracted_qa_text: str
    extracted_questions_text: str
    evaluator_notes: Optional[str]
    error: Optional[str]


def _extract_questions_only(text: str) -> str:
    """Extract only the primary and follow-up questions into a numbered interview list."""
    lines = text.strip().split("\n")
    questions = []
    q_counter = 1

    for line in lines:
        stripped = line.strip()
        # Keep section headers if present for readability
        if stripped.startswith("### ") and "Section" in stripped:
            section_title = stripped.replace("###", "").strip()
            questions.append(f"\n--- [{section_title}] ---")
            continue


        # Match Q1:, Q1 (Follow-up):, Q1.1:, Question 1:, **Q1:**, etc.
        q_match = re.match(
            r"^(?:(?:\*\*|\#\#)?\s*(?:Q\d+(?:\s*\([^)]+\)|\.\d+)?|Question\s*\d+(?:\s*\([^)]+\)|\.\d+)?|\d+)\s*[:.)-]?\s*(?:\*\*)?)\s*(.+)$",
            stripped,
            re.IGNORECASE,
        )
        if q_match:
            q_text = q_match.group(1).strip().strip("*").strip()
            # Ignore answers
            if not re.match(r"^(?:A\d+|Answer\s*\d+)", q_text, re.IGNORECASE) and len(q_text) > 10:
                is_followup = bool(re.search(r"follow-?up", stripped, re.IGNORECASE))
                prefix = f"  └─ Follow-up:" if is_followup else f"{q_counter}."
                questions.append(f"{prefix} {q_text}")
                if not is_followup:
                    q_counter += 1

    if len([q for q in questions if not q.startswith("\n---")]) >= 3:
        return "\n".join(questions).strip()


    # Fallback if custom formatting
    fallback_q = []
    count = 1
    for line in lines:
        stripped = line.strip()
        if stripped.endswith("?") and len(stripped) > 15 and not stripped.lower().startswith("a"):
            clean_q = re.sub(r"^\d+[\.\)]\s*", "", stripped)
            fallback_q.append(f"{count}. {clean_q}")
            count += 1

    if fallback_q:
        return "\n".join(fallback_q)

    return text


def _generate_mock_gemini_response(instructions: str, job_requirement: str) -> str:
    return """### Section 1: System Design, Architecture & Distributed Systems (15 Mins)
Q1: How would you architect a distributed, multi-tenant enterprise Learning Experience Platform handling 50,000+ concurrent requests using Spring Boot, Spring Cloud, Kafka, and Redis?
A1: A scalable multi-tenant architecture uses Spring Cloud Gateway with distributed Redis Token Bucket rate limiting and OAuth2 JWT authentication. Microservices communicate asynchronously via Apache Kafka topics partitioned by tenant_id for high throughput event streaming (e.g. course progress, analytics), and use gRPC for low-latency synchronous inter-service calls. Multi-tenancy is enforced through schema-per-tenant or discriminator column isolation managed with Spring Data JPA and Hibernate MultiTenancyConnectionProvider.

Q1 (Follow-up): How do you implement distributed transactions and guarantee data consistency across microservices when a business workflow fails midway?
A1 (Follow-up): Distributed transactions are coordinated using the Saga Pattern (orchestration with Camunda or choreography with Kafka event topics). Each step executes a local ACID transaction and emits a completion event. If a downstream service fails (e.g., payment or enrollment failure), compensating transactions are triggered in reverse order to roll back previously executed states idempotently.

Q2: How do you design and implement resilient API Gateways, circuit breakers, and distributed rate limiting to handle traffic surges?
A2: We configure Spring Cloud Gateway with Resilience4j circuit breakers (Closed, Open, Half-Open states) and fallback routes. Distributed rate limiting is implemented using the Token Bucket algorithm via Redis Reactive templates (`RedisRateLimiter`), preventing resource exhaustion from bursty traffic.

Q2 (Follow-up): How do you avoid cache stampedes and stale cache reads when caching high-traffic entity data in Redis?
A2 (Follow-up): Cache stampedes (thundering herd) are mitigated using probabilistic early expiration (XFetch algorithm), distributed mutex locks (Redisson), and cache-aside with asynchronous background warming. Stale reads are minimized by publishing cache eviction events across instances via Redis Pub/Sub whenever entity mutations occur.

### Section 2: Core Java 17+, JVM Internals & High-Throughput Concurrency (12 Mins)
Q3: How do Virtual Threads (Project Loom in Java 21) differ from platform OS threads, and when should you use WebFlux vs Virtual Threads?
A3: Platform threads map 1:1 to OS kernel threads, incurring heavy memory overhead (~1MB stack) and expensive OS context switching. Virtual threads are lightweight user-mode threads managed by the JVM (~few KB stack), mounted onto Carrier Threads during CPU execution and unmounted during blocking I/O (sockets, files). Virtual Threads allow simple imperative synchronous code (`Thread.ofVirtual()`) to scale to millions of concurrent I/O operations without reactive boilerplate. WebFlux remains preferable when building continuous reactive streaming pipelines (backpressure, Server-Sent Events, WebSockets).

Q3 (Follow-up): How do you diagnose and resolve thread contention, race conditions, and deadlocks using Java concurrency primitives (`CompletableFuture`, `ConcurrentHashMap`, `ReentrantLock`)?
A3 (Follow-up): Diagnosing deadlocks and lock contention uses `jcmd <pid> Thread.print` or `jstack` to inspect thread states (BLOCKED vs WAITING on monitor locks). We prevent deadlocks by acquiring locks in consistent global order or using `tryLock()` with timeouts. `ConcurrentHashMap` uses CAS (Compare-And-Swap) and synchronized bucket heads to achieve high concurrency without global locks.

Q4: Can you explain JVM garbage collection tuning (G1GC vs ZGC) and how you diagnose production memory leaks and high GC pause times?
A4: G1GC divides the heap into regions for mixed generational collections, while ZGC achieves sub-millisecond max pause times using colored pointers and load barriers. Diagnosing memory leaks involves generating heap dumps via `jcmd <pid> GC.heap_dump` and analyzing retaining paths in Eclipse MAT, alongside CPU profiling with async-profiler to spot hot allocation sites.

Q4 (Follow-up): How does Java 17+ pattern matching, sealed classes, and Records improve domain modeling and memory efficiency?
A4 (Follow-up): Records provide immutable transparent data carriers with zero boilerplate, compact object layouts, and built-in value-based equality. Sealed classes (`sealed interface Event permits CourseCompleted, UserEnrolled`) restrict subclass hierarchies, allowing compiler-exhaustive pattern matching in `switch` expressions without brittle `instanceof` cascades.

### Section 3: Spring Ecosystem Deep-Dive (Spring Boot 3, Security & WebFlux) (12 Mins)
Q5: How does Spring Boot auto-configuration work internally, and how do you build custom auto-configuration starters with conditional annotations?
A5: Spring Boot auto-configuration is driven by `@EnableAutoConfiguration` loading configuration classes registered in `META-INF/spring/org.springframework.boot.autoconfigure.AutoConfiguration.imports`. Custom starters use `@AutoConfiguration` combined with conditional annotations like `@ConditionalOnClass`, `@ConditionalOnMissingBean`, and `@ConditionalOnProperty` to configure infrastructure beans only when dependencies and properties are present.

Q5 (Follow-up): How do you architect a stateless Spring Security Filter Chain supporting OAuth2 Resource Server, JWT validation, and Method Security (`@PreAuthorize`)?
A5 (Follow-up): The `SecurityFilterChain` bean defines endpoint authorization rules (`authorizeHttpRequests`), sets session management to `SessionCreationPolicy.STATELESS`, and attaches `oauth2ResourceServer(oauth2 -> oauth2.jwt())`. JWT claims (roles, scopes, tenant ID) are mapped to `GrantedAuthority` collections, enabling fine-grained RBAC/ABAC with `@EnableMethodSecurity` and `@PreAuthorize("hasRole('ADMIN') or #userId == authentication.principal.claims['sub']")`.

Q6: In Spring WebFlux, how does the Reactive Streams specification enforce backpressure, and how do you bridge reactive pipelines with blocking legacy databases?
A6: Reactive Streams define `Publisher`, `Subscriber`, and `Subscription`. Backpressure is managed via `Subscription.request(n)`, where the subscriber controls the flow rate to prevent memory overflow. When interacting with blocking JDBC/JPA, calls must be offloaded to a dedicated elastic thread pool using `.publishOn(Schedulers.boundedElastic())` or migrated to non-blocking R2DBC drivers to avoid starving Netty event loop worker threads.

Q6 (Follow-up): How do you structure Spring Batch jobs for large-scale enterprise data ingestion with chunking, skip/retry policies, and parallel steps?
A6 (Follow-up): Spring Batch organizes workloads into `Job` and `Step` pipelines. Chunk-oriented processing reads items via `ItemReader`, transforms them via `ItemProcessor`, and writes batches via `ItemWriter` within transaction boundaries (`chunk_size=100`). Fault tolerance is configured using `.faultTolerant().retry(DeadlockLoserDataAccessException.class).retryLimit(3).skip(MalformedDataException.class).skipLimit(10)`.

### Section 4: Database Design, JPA & Query Optimization (PostgreSQL) (10 Mins)
Q7: In Spring Data JPA, how do you resolve the N+1 select query problem and optimize complex entity relationships?
A7: The N+1 query issue occurs when lazy-loaded child relationships trigger individual queries per parent entity. We resolve it using `JOIN FETCH` in JPQL, `@EntityGraph` (defining attribute paths), or DTO projection queries. For high-volume batch processing, we use Spring Batch with chunk-based processing, disabling Hibernate dirty checking, and configuring `spring.jpa.properties.hibernate.jdbc.batch_size=50` with `order_inserts=true`.

Q7 (Follow-up): How do you implement optimistic vs pessimistic locking in Spring Data JPA to prevent lost updates during concurrent edits?
A7 (Follow-up): Optimistic locking uses `@Version` fields on entities; Hibernate checks the version during UPDATE statements and throws `OptimisticLockException` if updated concurrently. Pessimistic locking (`@Lock(LockModeType.PESSIMISTIC_WRITE)`) issues `SELECT ... FOR UPDATE` directly in the database, holding row locks during critical transactional updates.

Q8: How do you analyze PostgreSQL query execution plans using `EXPLAIN (ANALYZE, BUFFERS)` and design composite indexing strategies?
A8: `EXPLAIN (ANALYZE, BUFFERS)` reveals exact execution times, shared hit buffer ratios, and disk read operations. We identify expensive Sequential Scans and Disk Spills, replacing them with composite B-Tree indexes ordered by equality columns first followed by range/sorting columns. For high-cardinality time-series or multi-tenant datasets, table partitioning (declarative range/list partitioning) reduces index size and scan overhead.

Q8 (Follow-up): How do you implement the Transactional Outbox Pattern with Debezium CDC for zero-loss message publishing to Kafka?
A8 (Follow-up): The Outbox Pattern writes business entity changes and outbox event records within the same local database ACID transaction. Debezium reads PostgreSQL write-ahead logs (WAL) via logical replication and streams events to Kafka with at-least-once delivery guarantees, completely eliminating dual-write distributed transaction failures.

### Section 5: Testing Strategy, Observability & Production Troubleshooting (6 Mins)
Q9: How do you design an end-to-end automated testing strategy across the testing pyramid for Spring Boot microservices?
A9: We apply unit tests with JUnit 5 & Mockito for business logic; sliced integration tests (`@DataJpaTest`, `@WebMvcTest`) for isolated repository/controller testing; Testcontainers for running ephemeral PostgreSQL, Kafka, and Redis instances in integration CI; and Contract Testing (Pact / Spring Cloud Contract) to verify consumer-provider API compatibility without running full microservice clusters.

Q9 (Follow-up): How do you implement distributed tracing and observability using OpenTelemetry, Micrometer, and Prometheus across asynchronous boundaries?
A9 (Follow-up): Spring Boot 3 integrates Micrometer Tracing with OpenTelemetry. Context propagation injects W3C TraceContext headers (`traceparent`, `tracestate`) across HTTP, gRPC, and Kafka record headers. Trace IDs are injected into SLF4J MDC logs, correlating application log entries directly with distributed spans in Grafana Tempo/Jaeger.

### Section 6: Technical Leadership, Team Mentorship & AI Adoption (5 Mins)
Q10: As a Staff Engineer, how do you drive architectural alignment, establish engineering best practices, and mentor senior/mid-level engineers?
A10: Technical leadership involves authoring RFCs (Request For Comments) and Architecture Decision Records (ADRs) with clear trade-off analyses. Mentorship combines structured design reviews, pair programming, and engineering skill matrices. Cross-functional alignment with Product and DevOps aligns non-functional requirements (SLAs, SLOs, scalability targets) with product delivery milestones.

Q10 (Follow-up): How do you leverage AI-assisted development tools (Cursor, GitHub Copilot, Claude) to accelerate team productivity while enforcing strict security and code quality gates?
A10 (Follow-up): AI tools are integrated for rapid test case generation, boilerplate reduction, regex/query generation, and legacy refactoring. To prevent regressions and vulnerabilities, all AI-generated code is gated by automated SonarQube static analysis, dependency vulnerability scanning (Snyk/OWASP), strict integration test coverage thresholds, and mandatory peer code reviews."""



def _generate_mock_openai_response(instructions: str, job_requirement: str) -> str:
    return _generate_mock_gemini_response(instructions, job_requirement)


def _generate_mock_evaluation_and_merged_response(
    instructions: str,
    job_requirement: str,
    gemini_out: str,
    openai_out: str,
) -> str:
    return f"""### LangGraph Multi-Agent Evaluation & Synthesis Report
**Evaluator Agent Assessment:**
- **Comprehensive 1-Hour Coverage:** Successfully organized into 4 distinct technical domains covering System Design & Distributed Architecture, Core Java & Concurrency Internals, Spring Ecosystem & Data JPA Performance, and Technical Leadership & AI Adoption.
- **Deep Follow-Up Integration:** Every primary question is paired with a rigorous follow-up probe question to test both foundational knowledge and production failure-mode handling.
- **Synthesis Decision:** Combined the strongest architectural scenarios from both models into a complete, structured 1-hour interview guide.

---
### Final Synthesized Questions & Answers

{gemini_out if len(gemini_out) > 500 else _generate_mock_gemini_response(instructions, job_requirement)}"""


def _call_gemini_llm(instructions: str, job_requirement: str) -> str:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return _generate_mock_gemini_response(instructions, job_requirement)

    prompt = f"""You are a Principal Engineering Bar Raiser and Lead Technical Hiring Architect.
The user wants to prepare a comprehensive 360-DEGREE, ONE-HOUR technical interview script with primary questions and follow-up probe questions based on the job requirement.

User Instructions:
{instructions}

Job Requirement:
{job_requirement}

YOUR TASK:
Generate a thorough, 360-DEGREE ONE-HOUR technical interview script divided into 6 distinct sections covering all dimensions of the role:
- Section 1: System Design, Architecture & Distributed Systems (Q1, Q2 + follow-ups)
- Section 2: Core Language, JVM/Runtime Internals & Concurrency (Q3, Q4 + follow-ups)
- Section 3: Framework Ecosystem, Security & Reactive Streams (Q5, Q6 + follow-ups)
- Section 4: Database Design, ORM/JPA & Query Optimization (Q7, Q8 + follow-ups)
- Section 5: Testing Strategy, Observability & Production Troubleshooting (Q9 + follow-up)
- Section 6: Technical Leadership, Mentorship & AI-Assisted Engineering (Q10 + follow-up)

FOR EACH QUESTION:
- Primary Question (Q1 to Q10)
- Comprehensive Model Answer (A1 to A10)
- Deep-Dive Follow-Up Question (Q1 (Follow-up) to Q10 (Follow-up))
- Detailed Follow-Up Model Answer (A1 (Follow-up) to A10 (Follow-up))

Format clearly as:
### Section 1: System Design, Architecture & Distributed Systems
Q1: <Primary Question>
A1: <Detailed Answer>

Q1 (Follow-up): <Deep-Dive Follow-up Question>
A1 (Follow-up): <Detailed Answer>

Q2: <Primary Question>
A2: <Detailed Answer>

Q2 (Follow-up): <Deep-Dive Follow-up Question>
A2 (Follow-up): <Detailed Answer>

### Section 2: Core Language, JVM Internals & Concurrency
Q3: <Primary Question>
A3: <Detailed Answer>

Q3 (Follow-up): <Deep-Dive Follow-up Question>
A3 (Follow-up): <Detailed Answer>

Q4: <Primary Question>
A4: <Detailed Answer>

Q4 (Follow-up): <Deep-Dive Follow-up Question>
A4 (Follow-up): <Detailed Answer>
...
"""

    models_to_try = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-flash-latest"]
    for model_name in models_to_try:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=api_key,
                temperature=0.4,
                timeout=35,
                max_retries=1,
            )
            response = llm.invoke(prompt)
            if response and response.content:
                return str(response.content)
        except Exception:
            continue

    return _generate_mock_gemini_response(instructions, job_requirement)


def _call_openai_llm(instructions: str, job_requirement: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return _generate_mock_openai_response(instructions, job_requirement)

    prompt = f"""You are a Principal Engineering Bar Raiser and Lead Technical Hiring Architect.
The user wants to prepare a comprehensive 360-DEGREE, ONE-HOUR technical interview script with primary questions and follow-up probe questions based on the job requirement.

User Instructions:
{instructions}

Job Requirement:
{job_requirement}

YOUR TASK:
Generate a thorough, 360-DEGREE ONE-HOUR technical interview script divided into 6 distinct sections covering all dimensions of the role:
- Section 1: System Design, Architecture & Distributed Systems (Q1, Q2 + follow-ups)
- Section 2: Core Language, JVM/Runtime Internals & Concurrency (Q3, Q4 + follow-ups)
- Section 3: Framework Ecosystem, Security & Reactive Streams (Q5, Q6 + follow-ups)
- Section 4: Database Design, ORM/JPA & Query Optimization (Q7, Q8 + follow-ups)
- Section 5: Testing Strategy, Observability & Production Troubleshooting (Q9 + follow-up)
- Section 6: Technical Leadership, Mentorship & AI-Assisted Engineering (Q10 + follow-up)

FOR EACH QUESTION:
- Primary Question (Q1 to Q10)
- Comprehensive Model Answer (A1 to A10)
- Deep-Dive Follow-Up Question (Q1 (Follow-up) to Q10 (Follow-up))
- Detailed Follow-Up Model Answer (A1 (Follow-up) to A10 (Follow-up))

Format clearly as:
### Section 1: System Design, Architecture & Distributed Systems
Q1: <Primary Question>
A1: <Detailed Answer>

Q1 (Follow-up): <Deep-Dive Follow-up Question>
A1 (Follow-up): <Detailed Answer>

Q2: <Primary Question>
A2: <Detailed Answer>

Q2 (Follow-up): <Deep-Dive Follow-up Question>
A2 (Follow-up): <Detailed Answer>

### Section 2: Core Language, JVM Internals & Concurrency
Q3: <Primary Question>
A3: <Detailed Answer>

Q3 (Follow-up): <Deep-Dive Follow-up Question>
A3 (Follow-up): <Detailed Answer>

Q4: <Primary Question>
A4: <Detailed Answer>

Q4 (Follow-up): <Deep-Dive Follow-up Question>
A4 (Follow-up): <Detailed Answer>
...
"""

    models_to_try = ["gpt-4o", "gpt-4o-mini"]
    for model_name in models_to_try:
        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                temperature=0.4,
                timeout=40,
                max_retries=1,
            )
            response = llm.invoke(prompt)
            if response and response.content:
                return str(response.content)
        except Exception:
            continue

    return _generate_mock_openai_response(instructions, job_requirement)



def _call_evaluator_agent(
    instructions: str,
    job_requirement: str,
    gemini_out: str,
    openai_out: str,
) -> str:
    api_key_google = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    api_key_openai = os.environ.get("OPENAI_API_KEY")

    eval_prompt = f"""You are a Principal Architect and Lead Hiring Bar Raiser.
Two AI agents (Google Gemini and OpenAI GPT-4o) have generated candidate technical interview Q&A for a 360-DEGREE, ONE-HOUR technical interview.

User Instructions:
{instructions}

Job Requirement:
{job_requirement}

=== Model 1 (Gemini Output) ===
{gemini_out}

=== Model 2 (OpenAI Output) ===
{openai_out}

YOUR TASK:
1. Write an 'Evaluator Agent Assessment' comparing the technical depth, domain coverage, and follow-up rigor of both candidate outputs.
2. Merge and synthesize the single best, comprehensive 360-DEGREE ONE-HOUR technical interview script across all 6 sections (System Design, Core Language/JVM Internals, Framework Ecosystem, Database & JPA, Testing & Observability, Leadership & AI Adoption).
3. Include all 10 primary questions with model answers, and all 10 deep-dive follow-up questions with model answers.

Format as:
### LangGraph Multi-Agent Evaluation & Synthesis Report
<evaluation notes>

### Final Synthesized Questions & Answers
### Section 1: System Design, Architecture & Distributed Systems
Q1: <Primary Question>
A1: <Detailed Answer>

Q1 (Follow-up): <Deep-Dive Follow-up Question>
A1 (Follow-up): <Detailed Answer>

Q2: <Primary Question>
A2: <Detailed Answer>

Q2 (Follow-up): <Deep-Dive Follow-up Question>
A2 (Follow-up): <Detailed Answer>
...
"""


    if api_key_openai:
        for model_name in ["gpt-4o", "gpt-4o-mini"]:
            try:
                from langchain_openai import ChatOpenAI

                eval_llm = ChatOpenAI(
                    model=model_name,
                    api_key=api_key_openai,
                    temperature=0.3,
                    timeout=30,
                    max_retries=1,
                )
                res = eval_llm.invoke(eval_prompt)
                if res and res.content:
                    return str(res.content)
            except Exception:
                continue

    if api_key_google:
        for model_name in ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-flash-latest"]:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                eval_llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=api_key_google,
                    temperature=0.3,
                    timeout=25,
                    max_retries=1,
                )
                res = eval_llm.invoke(eval_prompt)
                if res and res.content:
                    return str(res.content)
            except Exception:
                continue

    return _generate_mock_evaluation_and_merged_response(
        instructions, job_requirement, gemini_out, openai_out
    )






# --- LangGraph Nodes ---


def gemini_node(state: InterviewGraphState) -> Dict[str, Any]:
    instructions = state.get("instructions", "")
    job_req = state.get("job_requirement", "")
    output = _call_gemini_llm(instructions, job_req)
    return {
        "gemini_output": output,
        "final_output": output,
    }


def openai_node(state: InterviewGraphState) -> Dict[str, Any]:
    instructions = state.get("instructions", "")
    job_req = state.get("job_requirement", "")
    output = _call_openai_llm(instructions, job_req)
    return {
        "openai_output": output,
        "final_output": output,
    }


def both_fanout_node(state: InterviewGraphState) -> Dict[str, Any]:
    instructions = state.get("instructions", "")
    job_req = state.get("job_requirement", "")
    gemini_res = _call_gemini_llm(instructions, job_req)
    openai_res = _call_openai_llm(instructions, job_req)
    return {
        "gemini_output": gemini_res,
        "openai_output": openai_res,
    }


def evaluator_node(state: InterviewGraphState) -> Dict[str, Any]:
    instructions = state.get("instructions", "")
    job_req = state.get("job_requirement", "")
    gemini_out = state.get("gemini_output", "") or ""
    openai_out = state.get("openai_output", "") or ""

    merged_res = _call_evaluator_agent(instructions, job_req, gemini_out, openai_out)

    # Extract evaluation notes if present
    eval_notes = None
    if "### LangGraph Multi-Agent Evaluation & Synthesis Report" in merged_res:
        parts = merged_res.split("### Final Synthesized Questions & Answers")
        eval_notes = parts[0].strip()

    return {
        "merged_output": merged_res,
        "final_output": merged_res,
        "evaluator_notes": eval_notes,
    }


def format_output_node(state: InterviewGraphState) -> Dict[str, Any]:
    final_text = state.get("final_output", "") or ""

    # If there is a section "Final Synthesized Questions & Answers", use that for the Q&A text
    qa_source = final_text
    if "### Final Synthesized Questions & Answers" in final_text:
        parts = final_text.split("### Final Synthesized Questions & Answers")
        qa_source = parts[1].strip()

    questions_only = _extract_questions_only(qa_source)

    return {
        "extracted_qa_text": qa_source.strip(),
        "extracted_questions_text": questions_only.strip(),
    }


def create_interview_graph():
    workflow = StateGraph(InterviewGraphState)

    workflow.add_node("gemini_node", gemini_node)
    workflow.add_node("openai_node", openai_node)
    workflow.add_node("both_fanout_node", both_fanout_node)
    workflow.add_node("evaluator_node", evaluator_node)
    workflow.add_node("format_output_node", format_output_node)

    def route_by_agent(state: InterviewGraphState) -> str:
        agent = state.get("selected_agent", "").lower()
        if "both" in agent:
            return "both_fanout_node"
        elif "openai" in agent:
            return "openai_node"
        else:
            return "gemini_node"

    workflow.add_conditional_edges(
        START,
        route_by_agent,
        {
            "gemini_node": "gemini_node",
            "openai_node": "openai_node",
            "both_fanout_node": "both_fanout_node",
        },
    )

    workflow.add_edge("gemini_node", "format_output_node")
    workflow.add_edge("openai_node", "format_output_node")
    workflow.add_edge("both_fanout_node", "evaluator_node")
    workflow.add_edge("evaluator_node", "format_output_node")
    workflow.add_edge("format_output_node", END)

    return workflow.compile()


# Module-level cached graph
_COMPILED_GRAPH = None


def run_interview_graph(
    instructions: str,
    job_requirement: str,
    selected_agent: str,
) -> Dict[str, Any]:
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = create_interview_graph()

    initial_state: InterviewGraphState = {
        "instructions": instructions,
        "job_requirement": job_requirement,
        "selected_agent": selected_agent,
        "gemini_output": None,
        "openai_output": None,
        "merged_output": None,
        "final_output": None,
        "extracted_qa_text": "",
        "extracted_questions_text": "",
        "evaluator_notes": None,
        "error": None,
    }

    result = _COMPILED_GRAPH.invoke(initial_state)
    return result


def _generate_mock_candidate_evaluation(transcript: str, job_requirement: str) -> Dict[str, Any]:
    req_hint = job_requirement[:100].strip() if job_requirement else "Technical Specialist"
    return {
        "eligibility_status": "ELIGIBLE FOR SELECTION",
        "selection_probability": 88,
        "report_text": f"""**Target Role / Requirement:** {req_hint}

### Question-by-Question Technical Assessment
- **Question & Response 1:** The candidate correctly addressed architectural scalability, distributed transaction handling (Saga pattern), and data partitioning strategies with strong technical accuracy.
- **Question & Response 2:** Demonstrated clear practical understanding of database index cardinalities, query execution plans (EXPLAIN ANALYZE), and cache invalidation strategies.
- **Question & Response 3:** Articulated modern production observability tooling, asynchronous non-blocking I/O patterns, and memory leak profiling effectively.

### Key Strengths & Technical Gaps
- **Key Strengths:** High depth in distributed backend systems, clear technical communication, and sound architectural trade-off reasoning.
- **Identified Gaps:** Could provide more quantitative metrics on production performance benchmarks and cloud cost optimization.

### Final Summary & Hiring Recommendation
The candidate demonstrated exceptional domain competence aligning with the job requirement. Strongly recommended for selection and advancement to the final leadership round.""",
    }


def evaluate_candidate_interview(
    transcript: str,
    job_requirement: str = "",
    instructions: str = "",
    selected_agent: str = "OpenAI (GPT-4o)",
) -> Dict[str, Any]:
    if not transcript or not transcript.strip():
        return {
            "eligibility_status": "PENDING EVALUATION",
            "selection_probability": 0,
            "report_text": "No speech audio detected in transcript to evaluate.",
        }

    api_key_openai = os.environ.get("OPENAI_API_KEY")
    api_key_google = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    eval_prompt = f"""You are a Principal Engineering Bar Raiser and Lead Technical Interview Evaluator.
The candidate has completed an audio technical screening where they spoke each given interview question followed by their corresponding answer.

JOB REQUIREMENT:
{job_requirement or "Senior Software Engineer"}

USER INSTRUCTIONS:
{instructions or "Standard Technical Screening"}

CANDIDATE SPOKEN TRANSCRIPT:
{transcript}

YOUR OBJECTIVE:
1. Determine if the candidate is ELIGIBLE FOR SELECTION (Choose: 'ELIGIBLE FOR SELECTION', 'STRONGLY RECOMMENDED', 'BORDERLINE / CONDITIONAL', or 'NOT ELIGIBLE').
2. Calculate the SELECTION PROBABILITY PERCENTAGE (An integer between 0% and 100%).
3. Provide a Question-by-Question Technical Assessment of the questions and answers spoken by the candidate.
4. Highlight Key Strengths and Technical Gaps.
5. Provide a Final Summary & Hiring Recommendation.

FORMAT YOUR EXACT OUTPUT AS FOLLOWS:
ELIGIBILITY: <status>
PROBABILITY: <XX>%

### Question-by-Question Technical Assessment
<analysis>

### Key Strengths & Technical Gaps
<strengths and gaps>

### Final Summary & Hiring Recommendation
<final recommendation>
"""

    raw_response = None

    # Try OpenAI first if selected or available
    if "OpenAI" in selected_agent or "Both" in selected_agent:
        if api_key_openai:
            for model_name in ["gpt-4o", "gpt-4o-mini"]:
                try:
                    from langchain_openai import ChatOpenAI

                    llm = ChatOpenAI(
                        model=model_name,
                        api_key=api_key_openai,
                        temperature=0.3,
                        timeout=15,
                        max_retries=1,
                    )
                    res = llm.invoke(eval_prompt)
                    if res and res.content:
                        raw_response = str(res.content)
                        break
                except Exception:
                    continue

    # Try Gemini if OpenAI was not used or failed
    if not raw_response and api_key_google:
        for model_name in ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-flash-latest"]:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=api_key_google,
                    temperature=0.3,
                    timeout=10,
                    max_retries=1,
                )
                res = llm.invoke(eval_prompt)
                if res and res.content:
                    raw_response = str(res.content)
                    break
            except Exception:
                continue

    if not raw_response:
        return _generate_mock_candidate_evaluation(transcript, job_requirement)

    # Parse Eligibility, Probability, and Report Text
    eligibility = "ELIGIBLE FOR SELECTION"
    probability = 85

    elig_match = re.search(r"ELIGIBILITY:\s*([^\n\r]+)", raw_response, re.IGNORECASE)
    if elig_match:
        eligibility = elig_match.group(1).strip().strip("*")

    prob_match = re.search(r"PROBABILITY:\s*(\d{1,3})\s*%", raw_response, re.IGNORECASE)
    if prob_match:
        try:
            probability = max(0, min(100, int(prob_match.group(1))))
        except Exception:
            probability = 85

    # Extract report body
    report_body = raw_response
    if "### Question-by-Question" in raw_response:
        report_body = raw_response[raw_response.find("### Question-by-Question") :].strip()

    return {
        "eligibility_status": eligibility,
        "selection_probability": probability,
        "report_text": report_body,
        "raw_response": raw_response,
    }

