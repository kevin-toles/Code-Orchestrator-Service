#!/usr/bin/env python3
"""
Add 955 validated index terms to validated_term_filter.json and backfill to chapters.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

# Paths
VALIDATED_TERMS_FILE = Path("/Users/kevintoles/POC/Code-Orchestrator-Service/data/validated_term_filter.json")
JSON_TEXTS_DIR = Path("/Users/kevintoles/POC/llm-document-enhancer/workflows/pdf_to_json/output/textbooks_json")
METADATA_DIR = Path("/Users/kevintoles/POC/llm-document-enhancer/workflows/metadata_extraction/output")

# The 955 validated terms
NEW_TERMS = [
    "(cmdb)", "(mvcc)", "(sca)", "(swebok)", "2pc systems", "__missing__ method",
    "abnormal scenario indicators", "accidental complexity", "accumulating snapshot",
    "accumulating snapshots", "acquire-release ordering", "actor model", "addorder() method",
    "address translation", "aggregations", "agile development", "akf scale cube for databases",
    "alexa skills", "alias templates", "aliasing", "amdahl's law", "analytic model space",
    "analytics maturity", "apache avro schemas", "api gateways", "api support for change streams",
    "apigatewayapplication class", "application metrics", "application-based release patterns",
    "application-level", "appsync", "architectural structures", "argument deduction",
    "ariadne framework", "ascii", "asp.net", "assignment operator", "atam presentations",
    "atomic commit", "atomic grain data", "atomic operations", "audit dimension", "audit logging",
    "automated self-service deployments", "automated testing", "autonomy",
    "aws (amazon web services)", "aws redshift", "azure stream analytics", "azure synapse",
    "b-tree", "b-trees", "backpressure", "backward compatibility", "bad_alloc", "base classes",
    "batch data ingestion", "batch data processing", "batch update", "before c++11",
    "benefits of lambda functions", "bi applications", "bit-level operations",
    "blameless post-mortems", "blending knowledge and memory", "blobs", "blocking calls",
    "blue-green deployment", "bone animation keys", "branch coverage metric", "bridge tables",
    "broadcast hash joins", "broker pattern", "brokerless messaging", "brute-force techniques",
    "brute-force testing", "buffer overflow", "building images", "building resolvers for custom scalars",
    "building search indexes", "built-in module", "burp intruder", "bus matrix",
    "business process management", "bypassing filters", "bytearray type", "bytes type",
    "bytes versus str instances", "byzantine fault tolerance", "byzantine faults", "cache misses",
    "canary requests", "cap principle", "cap theorem", "capturing metadata", "cascading failures",
    "causal ordering", "cellular automata", "chain replication", "change data capture (cdc)",
    "change tracking", "chaos monkey", "chaos testing", "character models", "child processes",
    "choosing abstraction levels", "ci/cd pipeline", "circuit breaker pattern", "circuit breakers",
    "claim accumulating snapshot", "class templates", "classical school of unit testing",
    "clickstream data", "client credentials flow", "client recovery", "client-server model",
    "client-server pattern", "client-server systems", "clobs", "clone", "cloud collaboration tools",
    "cloud data platforms", "cloud key/value data stores", "cloud relational databases",
    "cloud warehouse", "cloud-scale service", "co-partitioning", "coalescing", "code coverage metric",
    "code generation", "code review systems", "cold backup", "combinational circuits", "commit list",
    "compared with inheritance", "compatibility rules", "compensating transaction", "compiling",
    "component-and-connector views", "composability problem", "composite keys",
    "compute-as-glue architecture", "concatenating", "concurrent programming", "condition codes",
    "conference application", "conflict detection", "conformed dimensions", "conformed facts",
    "consistent prefix reads", "consistent snapshots", "consumer-driven gateways",
    "container schedulers", "context switches", "context variables", "continuous delivery (cd)",
    "continuous integration (ci)", "conway's law", "copy-on-write", "core i7",
    "correctness of dataflow systems", "counterexamples", "crash recovery",
    "createordersaga orchestrator", "createordersagastate class", "creating buckets",
    "creating dataset reader", "creating lambda functions", "creating orchestration-based saga",
    "critical paths", "cross-channel timing dependencies", "cross-training", "cursor stability",
    "custom scalars", "customer dimension", "cyclomatic complexity", "cypher query language",
    "daily deployments", "dark launches", "data alignment", "data consumers", "data cubes",
    "data deduplication", "data encryption", "data generation", "data hazards",
    "data highway planning", "data integration", "data loading", "data mining", "data movement",
    "data platform storage", "data processing and manipulation", "data quality checks",
    "data sources", "data strategy", "data type conversion", "data types", "data warehousing",
    "database log", "database system", "datacenters", "dataframe method", "datanodes",
    "date dimension", "date/time dimensions", "deadlocks", "deal dimension", "decentralization",
    "declaring", "decommissioning", "dedicated release engineer", "deduction guide",
    "deduplication system", "defining aggregate commands", "degenerate dimension",
    "degenerate dimensions", "delegation", "demand paging", "dependency graph", "dependency scanning",
    "deployment flow", "deployment frequency", "deployment lead time", "design for operations",
    "detailed dimension model", "detecting concurrent writes", "developing in java ee",
    "developing lambda functions", "dimension tables", "dimensional modeling",
    "direct data platform access", "direct-mapped caches", "dirty page table", "disaster preparedness",
    "distributed state", "distributed transaction support", "document data model",
    "document embeddings", "document-partitioned indexes", "domain modeling", "domain name",
    "domain-driven design (ddd)", "domain-specific", "dora metrics", "doubles", "downscaling",
    "drawbacks of lambda functions", "drill across", "drill down", "drilling down",
    "dsl (domain-specific language)", "duplicate messages", "durations", "dynamic content",
    "dynamic import", "dynamic memory allocation", "dynamic partitioning", "eclipse",
    "encoder-decoder attention", "encodings", "end-to-end tests", "enterprise java bean (ejb)",
    "enterprise service bus (esb)", "enterprise service buses (esbs)", "enumerations",
    "environment development on demand", "environment settings",
    "environments stored in version control", "equi-joins", "error budgets", "error conditions",
    "error level", "error messages", "ethernet", "etl development",
    "etl overlay and metadata repository", "etl system", "etl systems", "event enrichment",
    "event processing", "event publishing", "event time versus processing time",
    "event-driven pipeline", "eventual consistency", "eventuate local event store",
    "eventuate tram saga framework", "evolutionary architecture", "evolvability",
    "exactly-once semantics", "example contract", "exception handler", "exception specifications",
    "exit status", "expiration", "expiration date", "explicit", "extensibility",
    "external cloud provider tools", "extract audio lambda function", "fact extractor",
    "fact tables", "factless fact tables", "fallbacks", "fan-out", "feature flags",
    "feature keyword", "fences", "few-shot prompting", "file format conversion", "flask-smorest",
    "forcing functions", "foreign keys", "formatting strings", "free lists", "front-end programs",
    "frozenset type", "ftp fragment", "full specialization", "full table ingestion",
    "full text search services", "fully associative caches", "function shipping",
    "functional architecture", "functional requirements", "functional units", "g/l (general ledger)",
    "ga (google analytics)", "garbage collection", "gauges", "gauntlt", "generator expressions",
    "givens", "global development", "gmt (greenwich mean time)", "google app engine",
    "google bigquery", "gossip protocol", "gpt-2 (generative pretraining)", "graceful degradation",
    "grain", "grains", "graphql api testing", "greedy path", "hacker's methodology",
    "handling http requests", "handling query parameters", "handling skew", "handshakes",
    "happens-before relationship", "happens-before relationships", "hardware architecture",
    "hash indexes", "hash joins", "hash partitioning", "hazards", "header/line patterns",
    "heterogeneity", "heterogeneous products", "heuristics", "hexagonal architecture",
    "hierarchies", "high availability", "high-level architecture", "histograms", "hot backup",
    "hot spots", "http-native apis with rest", "hypertext transfer protocol (http)", "i/o devices",
    "iaas (infrastructure as a service)", "idempotence", "idempotentupdate() method",
    "identity and access management", "implemented in smalltalk", "implementing abort",
    "implementing api endpoints", "implementing commit", "implementing endpoints",
    "implementing field resolvers", "implementing mutation resolvers", "implementing query resolvers",
    "implementing type resolvers", "ims tm", "in batch processing", "in-memory cache",
    "in-memory databases", "in-memory storage", "incremental development", "incremental processing",
    "incremental table ingestion", "information radiator", "infrastructure as a service (iaas)",
    "init capture", "initializer lists", "initializing web application", "instances and fields",
    "instruction processing", "integers", "integral data types", "integration testing",
    "integration tests", "interaction styles", "interactive shell",
    "interface definition language (idl)", "interlocks", "internet of things",
    "interpositioning libraries", "intrusion detection", "invariant", "invariants",
    "inverse document frequency (idf)", "invocations", "invoking lambda functions",
    "isolating tests", "issue tracking systems", "iterative processing", "java ee",
    "java transaction api (jta)", "job queues", "json module", "json-rpc", "junk dimensions",
    "k-nearest neighbors", "key-range partitioning", "key-value stores", "keyword-only arguments",
    "kinesis data analytics", "kitchenserviceproxy class", "ktables", "kubernetes resources",
    "l2 regularization", "lambda architecture", "lambda expression", "lambda role arn",
    "late binding", "layered pattern", "laying out project structure", "ldap injection",
    "leader-based replication", "leaderless replication", "learning culture", "leases",
    "left-outer join", "legacy api proxy", "levels of instantiation",
    "lightweight architecture evaluation", "linux/x86-64 systems", "lock managers", "lock table",
    "lock-adopting constructors", "lock-ins", "locking granularity", "log compaction",
    "log sequence number", "log-based", "log-structured storage", "logic gates", "logical clock",
    "logical operations", "london school of unit testing", "machine language", "machine learning (ml)",
    "machine translation (mt)", "machine-language procedures", "machine-level data",
    "machine-level programming", "maintaining data consistency", "maintaining derived state",
    "making work visible", "many-to-one hierarchies", "map-reduce pattern", "map-reduce patterns",
    "marshaling", "massively parallel processing (mpp)", "master nodes", "materialization",
    "materialized views", "maximizing test accuracy", "mean time between failures (mtbf)",
    "mean time to repair", "mean time to repair (mttr)", "measure type dimension", "media services",
    "member templates", "memory blocks", "memory technology", "memory-mapped i/o",
    "merge video lambda function", "merge-join", "message bus architectures", "message ordering",
    "message-oriented middleware", "metadata api", "metadata database", "metrics sources",
    "micro-frontends", "microbatching", "midjourney", "mini-dimensions", "mirrored disks",
    "mockito", "model-view-controller pattern", "modeling data", "monoliths", "monotonic reads",
    "moore's law", "move constructors", "move-assignment operators", "mttr (mean time to repair)",
    "multi-datacenter support", "multi-tier pattern", "multigranularity locking", "multitasking",
    "multitenant systems", "multivalued", "multivalued dimensions", "mutual exclusion", "n-grams",
    "name-value pairs", "naming conventions", "naming standards", "natural keys",
    "need for distributed transactions", "network faults", "network latency", "neural networks",
    "nonlocal jumps", "nonrepeatable reads", "not null", "numeric attributes", "numeric facts",
    "numpy array vectorized version", "numpy ndarrays", "object type", "object-oriented",
    "object-oriented design", "observable services", "observing derived state", "offsets",
    "on-call support", "online gaming use case", "operands", "operational product master",
    "operator<<", "operator=", "operator>", "operator[]", "opportunity/stakeholder matrix",
    "optimistic concurrency control", "optimistic locking", "optimistic updates", "order aggregate",
    "order transactions", "ordercommandhandlers class", "orderconfiguration class",
    "orderhandlers class", "orderservice class", "orderserviceconfiguration class", "os threads",
    "os.path module", "osgi alliance", "outer joins", "outriggers", "oversubscription",
    "paas (platform as a service)", "pack expansion", "packaged data models",
    "packaging service as zip file", "palm method", "parallel processing", "parallel quicksort",
    "parquet", "partial specialization", "partitioned hash joins", "path length", "peer review",
    "peer-to-peer pattern", "peer-to-peer systems", "perfect forwarding", "performance analysis",
    "periodic snapshot", "periodic snapshots", "persistence integration tests", "persistent storage",
    "persisting aggregates using events", "personas", "php api methods", "ping/echo",
    "pipe-and-filter pattern", "pipelining", "pkce flow", "platform as a service (paas)",
    "point-of-sale systems", "port forwarding", "portability", "pos (part-of-speech) tagging",
    "positional-only arguments", "pre-web era", "prefetching", "presentation area", "presumed abort",
    "pretraining", "preventing lost updates", "preventing write skew", "problem breakdown",
    "problems with hash mod n", "process group", "process pauses", "processor context",
    "producer properties", "producers", "product dimension", "production metrics", "progress graphs",
    "project structure", "property graphs", "protocol (amqp)", "protocol selection", "prototypes",
    "prototyping", "pseudo-conversations", "psql shell", "publicizing post-mortems",
    "publish-subscribe", "publish-subscribe pattern", "qa engineers", "quadtrees", "quality screens",
    "query optimization", "query optimizer", "query processing", "queryable state", "race conditions",
    "range partitioning", "rate limits", "read committed isolation", "read locks", "read quorum",
    "read uncommitted", "read-only transactions", "readability", "real-time analytics",
    "real-time data processing and analytics", "real-time processing", "real-time processing services",
    "real-time storage", "real-time systems", "reaping", "record stream", "recovery manager",
    "recursive", "recursive dimensional clustering (rdc)", "recursive query support",
    "recursive sql queries", "referential integrity", "registering classes", "regression tests",
    "relational algebra and sql", "relfrozenxid of table", "remote procedure call",
    "reordering imports", "replacement policies", "replication controller creation",
    "replication controller deletion", "representational state transfer (rest)", "reprocessing data",
    "request routing", "resilience engineering", "resource management",
    "rest (representational state transfer)", "retail use case", "return statement",
    "reverse proxy service", "risc design", "risc processors", "role playing", "rolling back",
    "rolling upgrades", "root exceptions for apis", "rpc systems", "ruby on rails", "rule-based",
    "run-time stack", "running as services", "running containers", "running default test suite",
    "savepoints", "scalability (hotspots)", "scalability (number of open tasks)", "scaling out",
    "scaling up", "scds (slowly changing dimensions)", "scenario line", "scheduled services",
    "schema evolution", "search engine", "security vulnerability", "self-healing",
    "self-service metrics", "self-service platforms", "semantic functions", "semantic memory",
    "semi-additive facts", "semijoin operation", "sequential processing",
    "sequentially consistent ordering", "serializability", "serializable isolation", "server recovery",
    "service health", "service level performance", "service meshes",
    "service oriented architecture (soa)", "service platform selection", "service proxy",
    "service-oriented architecture", "service-oriented architecture pattern", "set associative caches",
    "set elements", "shadow maps", "shadow paging", "shared nothing", "shared state",
    "shared-data pattern", "shift operations", "short-circuiting", "shrunken rollup dimensions",
    "side effects", "silverlight", "simian army", "simulations", "single customer dimension",
    "single responsibility principle", "single-threaded execution", "skill keywords", "sloppy quorums",
    "sloppy quorums and hinted handoff", "smart keys", "smoke testing", "snapshot isolation support",
    "socket function", "software raid", "source code integrity", "source code repository",
    "specifying prerequisites with givens", "specifying rest apis",
    "split and convert video lambda function", "spring framework", "sprints",
    "sql server service broker", "stable storage", "stack pointers", "staging environments",
    "star schemas", "state changes", "state machine replication", "state machines",
    "state management", "stateless servers", "statement-based replication", "static libraries",
    "statistical analysis", "statistical and numerical algorithms", "std::add_lvalue_reference",
    "std::execution::sequenced_policy", "std::remove_const", "std::remove_reference", "std::vector",
    "step dimension", "stopiteration exception", "stored procedures", "str.format method",
    "stream joins", "streaming data ingestion", "streams and event processing", "striding", "stripe",
    "struct module", "structured logging", "stubs", "submissions queue", "subprocess module",
    "subsystems", "subtransactions", "subtype", "subtypes", "subword information", "subword models",
    "successor", "summary statistics", "super-peer", "surrogate keys", "swap space", "symbol tables",
    "system architecture", "system calls", "system management", "systems of record", "table creation",
    "table-testing criteria", "technical heterogeneity", "technical metadata layer",
    "term frequency (tf)", "termination protocol", "test types", "test-driven development",
    "testing environments", "text mining", "text parsers", "text string", "textual facts",
    "tf-idf calculation", "third-party", "thomas write rule", "threading module", "threat landscape",
    "three-tier architecture", "three-tier web service", "thrift and protocol buffers",
    "ticket aggregate", "time module", "time points", "time-of-day", "timeout period",
    "timestamp ordering", "tokenization", "tombstone", "train-serve skew", "training model",
    "training pipeline", "transaction counts", "transaction handshakes", "transaction management",
    "transaction numbers", "transaction processing", "transactional messaging", "transactional rpc",
    "transcode video lambda function", "transcoding video", "transformer as language model",
    "transformer-xl", "transport layer security (tls)", "tuple type", "tuxedo", "tuxedo (oracle)",
    "two-phase", "two-phase commit", "two-phase commit (2pc)", "two-phase locking (2pl)",
    "two-pizza team", "type 1 in same dimension", "type 2 in same dimension", "typedef",
    "types of windows", "ubuntu instance", "unions", "upcasting", "upsampling and downsampling",
    "usability testing", "use of adapter", "use of bridge", "use of chain of responsibility",
    "use of command", "use of composite", "use of facade", "use of factory method",
    "use of interpreter", "use of iterator", "use of memento", "use of observer", "use of zookeeper",
    "user input", "user-defined", "using dapr components", "using mocks to verify behavior",
    "utc (coordinated universal time)", "validating payloads with unknown fields",
    "validating request payloads with pydantic", "validating subclasses",
    "validating url query parameters", "value category", "variability", "vector similarity search",
    "version control systems", "version vectors", "virtual environments", "virtual machines (vms)",
    "vms (virtual machines)", "vocabulary and token indexers", "void*", "warm backup",
    "weighting", "weighting losses", "wer (windows error reporting)", "white-box testing",
    "whitespace", "with pydantic", "with rdbms-based event store", "word associations",
    "work in progress (wip)", "work stealing", "write quorum", "write skew",
    "write skew (transaction isolation)", "ws-transactions", "xa interface", "xa transactions",
    "xml-rpc", "xpath", "y86-64 pipelining", "zero initialization", "zero-copy interactions",
    "zero-downtime deployments", "zero-shot cot prompting", "fine-grained", "flash memory"
]


def find_index_pages(pages: list) -> str:
    """Find and concatenate index/glossary pages."""
    index_content = []
    in_index = False
    
    for page in pages:
        content = page.get("content", "") if isinstance(page, dict) else str(page)
        if not in_index:
            first_100 = content[:100].lower()
            if any(marker in first_100 for marker in ['index\n', 'index\r', 'glossary\n']):
                in_index = True
        if in_index:
            index_content.append(content)
            if len(index_content) > 2:
                first_50 = content[:50].lower()
                if any(end in first_50 for end in ['other books', 'about the author', 'colophon']):
                    break
    return "\n".join(index_content)


def parse_index_entries(index_content: str) -> list[dict]:
    """Parse index entries from content. Returns list of {term, pages}."""
    entries = []
    patterns = [
        r'^(.+?)\s{2,}(\d[\d,\s\-–n]+)$',
        r'^([^,]+),\s*(\d[\d,\s\-–n]+)$',
        r'^([a-zA-Z][\w\s\(\)\-]+?)\s+(\d{1,4}(?:[\s,\-–n]+\d{1,4})*)$',
        r'^(.+?)\s+(\d{1,4}(?:[\s,\-–n]+\d{1,4})*)$',
    ]
    
    for line in index_content.split('\n'):
        line = line.strip()
        if not line or len(line) <= 2 or line.isdigit():
            continue
        
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                term = match.group(1).strip()
                pages_str = match.group(2).strip()
                
                if re.search(r'[a-zA-Z]{2,}', term) and re.search(r'\d', pages_str):
                    term = re.sub(r'\s+', ' ', term)
                    term = re.sub(r'^[•\-\*]\s*', '', term)
                    
                    if 3 <= len(term) <= 80:
                        # Parse page numbers
                        pages = []
                        pages_str = re.sub(r'n(?=\s|,|$)', '', pages_str)
                        for part in re.split(r'[,\s]+', pages_str):
                            part = part.strip()
                            if '-' in part or '–' in part:
                                range_parts = re.split(r'[-–]', part)
                                if len(range_parts) == 2:
                                    try:
                                        start, end = int(range_parts[0]), int(range_parts[1])
                                        if end > start and (end - start) < 100:
                                            pages.extend(range(start, end + 1))
                                    except ValueError:
                                        pass
                            else:
                                try:
                                    pages.append(int(part))
                                except ValueError:
                                    pass
                        
                        if pages:
                            entries.append({"term": term, "pages": pages})
                    break
    return entries


def map_pages_to_chapters(entries: list[dict], chapters: list[dict]) -> dict:
    """Map index entries to chapters based on page numbers."""
    chapter_terms = defaultdict(set)
    
    page_to_chapter = {}
    for i, ch in enumerate(chapters):
        start = ch.get("start_page", 0)
        end = ch.get("end_page", 0)
        if start and end:
            for p in range(start, end + 1):
                page_to_chapter[p] = i
    
    for entry in entries:
        term = entry["term"]
        for page in entry["pages"]:
            if page in page_to_chapter:
                chapter_idx = page_to_chapter[page]
                chapter_terms[chapter_idx].add(term)
    
    return chapter_terms


def main():
    print("=" * 70)
    print("ADD INDEX TERMS AND BACKFILL")
    print("=" * 70)
    
    # Step 1: Add terms to validated_term_filter.json
    print("\n1. Adding terms to validated_term_filter.json...")
    
    with open(VALIDATED_TERMS_FILE) as f:
        validated = json.load(f)
    
    existing_keywords = set(k.lower() for k in validated.get("keywords", []))
    existing_concepts = set(c.lower() for c in validated.get("concepts", []))
    all_existing = existing_keywords | existing_concepts
    
    print(f"   Existing: {len(all_existing):,} terms")
    
    truly_new = [t for t in NEW_TERMS if t.lower() not in all_existing]
    print(f"   New terms to add: {len(truly_new)}")
    
    validated["concepts"].extend(truly_new)
    validated["concepts"] = sorted(set(validated["concepts"]))
    
    with open(VALIDATED_TERMS_FILE, 'w') as f:
        json.dump(validated, f, indent=2)
    
    print(f"   Updated total: {len(validated['keywords']) + len(validated['concepts']):,} terms")
    
    # Create set of new terms for matching
    new_terms_set = set(t.lower() for t in NEW_TERMS)
    
    # Step 2: Backfill to chapters using index page-to-chapter mapping
    print("\n2. Backfilling terms to chapters...")
    
    stats = {
        "books_processed": 0,
        "books_updated": 0,
        "chapters_updated": 0,
        "terms_added": 0
    }
    
    json_files = sorted(JSON_TEXTS_DIR.glob("*.json"))
    print(f"   Processing {len(json_files)} books...")
    
    for json_path in json_files:
        book_name = json_path.stem
        stats["books_processed"] += 1
        
        # Find metadata file
        metadata_path = METADATA_DIR / f"{book_name}_metadata.json"
        if not metadata_path.exists():
            candidates = list(METADATA_DIR.glob(f"*{book_name[:20]}*_metadata.json"))
            if candidates:
                metadata_path = candidates[0]
            else:
                continue
        
        # Load book JSON
        try:
            with open(json_path) as f:
                book_data = json.load(f)
        except:
            continue
        
        pages = book_data.get("pages", [])
        if not pages:
            continue
        
        # Parse index
        index_content = find_index_pages(pages)
        if not index_content:
            continue
        
        entries = parse_index_entries(index_content)
        if not entries:
            continue
        
        # Filter to only new terms
        filtered_entries = []
        for entry in entries:
            if entry["term"].lower() in new_terms_set:
                filtered_entries.append(entry)
        
        if not filtered_entries:
            continue
        
        # Load metadata
        with open(metadata_path) as f:
            chapters = json.load(f)
        
        # Map to chapters
        chapter_terms = map_pages_to_chapters(filtered_entries, chapters)
        
        # Update chapters
        book_modified = False
        for ch_idx, terms in chapter_terms.items():
            if ch_idx >= len(chapters):
                continue
            
            chapter = chapters[ch_idx]
            existing = set(k.lower() for k in chapter.get("keywords", []))
            
            new_terms_for_chapter = [t for t in terms if t.lower() not in existing]
            
            if new_terms_for_chapter:
                if "keywords" not in chapter:
                    chapter["keywords"] = []
                chapter["keywords"].extend(new_terms_for_chapter)
                stats["terms_added"] += len(new_terms_for_chapter)
                stats["chapters_updated"] += 1
                book_modified = True
        
        if book_modified:
            with open(metadata_path, 'w') as f:
                json.dump(chapters, f, indent=2)
            stats["books_updated"] += 1
    
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Books processed: {stats['books_processed']}")
    print(f"Books updated: {stats['books_updated']}")
    print(f"Chapters updated: {stats['chapters_updated']}")
    print(f"Terms added: {stats['terms_added']}")


if __name__ == "__main__":
    main()
