#!/usr/bin/env python3
"""Complete verification of ALL 955 KEEP terms."""

import json
from pathlib import Path

# ALL 955 KEEP terms from the user's list
KEEP_TERMS = [
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
    "automated self-service deployments", "automated testing", "autonomy", "aws (amazon web services)",
    "aws redshift", "azure stream analytics", "azure synapse", "b-tree", "b-trees", "backpressure",
    "backward compatibility", "bad_alloc", "base classes", "batch data ingestion",
    "batch data processing", "batch update", "before c++11", "benefits of lambda functions",
    "bi applications", "bit-level operations", "blameless post-mortems",
    "blending knowledge and memory", "blobs", "blocking calls", "blue-green deployment",
    "bone animation keys", "branch coverage metric", "bridge tables", "broadcast hash joins",
    "broker pattern", "brokerless messaging", "brute-force techniques", "brute-force testing",
    "buffer overflow", "building images", "building resolvers for custom scalars",
    "building search indexes", "built-in module", "burp intruder", "bus matrix",
    "business process management", "bypassing filters", "bytearray type", "bytes type",
    "bytes versus str instances", "byzantine fault tolerance", "byzantine faults", "cache misses",
    "canary requests", "cap principle", "cap theorem", "capturing metadata", "cascading failures",
    "causal ordering", "cellular automata", "chain replication", "change data capture (cdc)",
    "change tracking", "change, implementing", "chaos monkey", "chaos testing", "character models",
    "child processes", "choosing abstraction levels", "ci/cd pipeline", "circuit breaker pattern",
    "circuit breakers", "claim accumulating snapshot", "class templates",
    "classical school of unit testing", "clickstream data", "client credentials flow",
    "client recovery", "client-server model", "client-server pattern", "client-server systems",
    "clobs", "clone", "cloud collaboration tools", "cloud data platforms",
    "cloud key/value data stores", "cloud relational databases", "cloud warehouse",
    "cloud-scale service", "co-partitioning", "coalescing", "code coverage metric",
    "code generation", "code review systems", "cold backup", "combinational circuits",
    "commit list", "compared with inheritance", "compatibility rules", "compensating transaction",
    "compiling", "component-and-connector views", "composability problem", "composite keys",
    "compute-as-glue architecture", "concatenating", "concurrent programming", "condition codes",
    "conference application", "conflict detection", "conformed dimensions", "conformed facts",
    "consistent prefix reads", "consistent snapshots", "consumer-driven gateways",
    "container schedulers", "content-length:", "context switches", "context variables",
    "continuous delivery (cd)", "continuous integration (ci)", "conway's law", "copy-on-write",
    "core i7", "correctness of dataflow systems", "counterexamples", "crash recovery",
    "createordersaga orchestrator", "createordersagastate class", "creating buckets",
    "creating dataset reader", "creating lambda functions", "creating orchestration-based saga",
    "critical paths", "cross-channel timing dependencies", "cross-training", "cursor stability",
    "custom scalars", "customer dimension", "cyclomatic complexity", "cypher query language",
    "daily deployments", "dark launches", "data alignment", "data consumers", "data cubes",
    "data deduplication", "data encryption", "data generation", "data hazards",
    "data highway planning", "data integration", "data loading", "data mining", "data movement",
    "data platform storage", "data processing and manipulation", "data quality checks",
    "data sources", "data strategy", "data type conversion", "data types", "data warehousing",
    "data, inspecting", "database log", "database system", "datacenters", "dataframe method",
    "datanodes", "date dimension", "date/time dimensions", "deadlocks", "deal dimension",
    "decentralization", "declaring", "decommissioning", "dedicated release engineer",
    "deduction guide", "deduplication system", "defining aggregate commands",
    "degenerate dimension", "degenerate dimensions", "delegation", "demand paging",
    "dependency graph", "dependency scanning", "deployment flow", "deployment frequency",
    "deployment lead time", "description and interests", "design for operations",
    "design of restaurant service", "detailed dimension model", "detecting concurrent writes",
    "developing in java ee", "developing lambda functions", "devops cafe podcast",
    "dimension tables", "dimensional modeling", "direct data platform access",
    "direct-mapped caches", "dirty page table", "disaster preparedness", "distributed state",
    "distributed transaction support", "document data model", "document embeddings",
    "document-partitioned indexes", "domain modeling", "domain name", "domain-driven design (ddd)",
    "domain-specific", "domain-specific examples", "dora metrics", "doubles", "downscaling",
    "drawbacks of lambda functions", "drill across", "drill down", "drilling down",
    "dsl (domain-specific language)", "duplicate messages", "durations", "dynamic content",
    "dynamic import", "dynamic memory allocation", "dynamic partitioning", "eclipse",
    "employee profiles", "encoder-decoder attention", "encodings", "end-to-end tests",
    "enterprise java bean (ejb)", "enterprise service bus (esb)", "enterprise service buses (esbs)",
    "enumerations", "environment development on demand", "environment settings",
    "environments stored in version control", "equi-joins", "error budgets", "error conditions",
    "error level", "error messages", "establishing", "ethernet", "etl development",
    "etl overlay and metadata repository", "etl system", "etl systems", "event enrichment",
    "event processing", "event publishing", "event time versus processing time",
    "event-driven pipeline", "eventual consistency", "eventuate local event store",
    "eventuate tram saga framework", "evolutionary architecture", "evolvability",
    "exactly-once semantics", "example contract", "exception handler", "exception specifications",
    "executable object files", "exit status", "expiration", "expiration date", "explicit",
    "extensibility", "external cloud provider tools", "extract audio lambda function",
    "fact extractor", "fact tables", "factless fact tables", "fallbacks", "fan-out",
    "feature flags", "feature keyword", "fences", "few-shot prompting", "file format conversion",
    "flask-smorest", "forcing functions", "foreign keys", "formatting strings", "free lists",
    "front-end programs", "frozenset type", "ftp fragment", "full specialization",
    "full table ingestion", "full text search services", "fully associative caches",
    "function shipping", "functional architecture", "functional requirements", "functional units",
    "g/l (general ledger)", "ga (google analytics)", "garbage collection", "gauges", "gauntlt",
    "generator expressions", "givens", "global development", "gmt (greenwich mean time)",
    "google app engine", "google bigquery", "gossip protocol", "gpt-2 (generative pretraining)",
    "graceful degradation", "grain", "grains", "graphql api testing", "greedy path",
    "hacker's methodology", "handling http requests", "handling query parameters", "handling skew",
    "handshakes", "happens-before relationship", "happens-before relationships",
    "hardware architecture", "hash indexes", "hash joins", "hash partitioning", "hazards",
    "header/line patterns", "heterogeneity", "heterogeneous products", "heuristics",
    "hexagonal architecture", "hierarchies", "high availability", "high-level architecture",
    "histograms", "hot backup", "hot spots", "http-native apis with rest",
    "hypertext transfer protocol (http)", "i/o devices", "iaas (infrastructure as a service)",
    "idempotence", "idempotentupdate() method", "identity and access management",
    "implemented in smalltalk", "implementing abort", "implementing api endpoints",
    "implementing commit", "implementing endpoints", "implementing field resolvers",
    "implementing mutation resolvers", "implementing query resolvers", "implementing type resolvers",
    "ims tm", "in batch processing", "in-memory cache", "in-memory databases", "in-memory storage",
    "incremental development", "incremental processing", "incremental table ingestion",
    "information radiator", "infrastructure as a service (iaas)", "init capture",
    "initializer lists", "initializing web application", "instances and fields",
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
    "left-outer join", "legacy api proxy", "less or equal", "levels of instantiation",
    "lightweight architecture evaluation", "linux/x86-64 systems", "lock managers", "lock table",
    "lock-adopting constructors", "lock-ins", "locking granularity", "log compaction",
    "log sequence number", "log-based", "log-structured storage", "logic gates", "logical clock",
    "logical operations", "london school of unit testing", "machine language",
    "machine learning (ml)", "machine translation (mt)", "machine-language procedures",
    "machine-level data", "machine-level programming", "maintaining data consistency",
    "maintaining derived state", "maintenance notes", "making work visible",
    "many-to-one hierarchies", "map-reduce pattern", "map-reduce patterns", "marshaling",
    "massively parallel processing (mpp)", "master nodes", "materialization", "materialized views",
    "maximizing test accuracy", "mean time between failures (mtbf)", "mean time to repair",
    "mean time to repair (mttr)", "measure type dimension", "media services", "member templates",
    "memory blocks", "memory technology", "memory-mapped i/o", "merge video lambda function",
    "merge-join", "message bus architectures", "message ordering", "message-oriented middleware",
    "metadata api", "metadata database", "metrics sources", "micro-frontends", "microbatching",
    "midjourney", "mini-dimensions", "mirrored disks", "mockito", "model-view-controller pattern",
    "modeling data", "monoliths", "monotonic reads", "moore's law", "move constructors",
    "move-assignment operators", "mttr (mean time to repair)", "multi-datacenter support",
    "multi-tier pattern", "multigranularity locking", "multitasking", "multitenant systems",
    "multivalued", "multivalued dimensions", "mutual exclusion", "n-grams", "name-value pairs",
    "naming conventions", "naming standards", "natural keys", "need for distributed transactions",
    "network faults", "network latency", "neural networks", "nonlocal jumps", "nonrepeatable reads",
    "not null", "numeric attributes", "numeric facts", "numpy array vectorized version",
    "numpy ndarrays", "object type", "object-oriented", "object-oriented design",
    "observable services", "observing derived state", "offsets", "on-call support",
    "online gaming use case", "operand specifiers", "operands", "operational product master",
    "operator<<", "operator=", "operator>", "operator[]", "opportunity/stakeholder matrix",
    "optimistic concurrency control", "optimistic locking", "optimistic updates", "order aggregate",
    "order transactions", "ordercommandhandlers class", "orderconfiguration class",
    "orderhandlers class", "orderservice class", "orderserviceconfiguration class", "os threads",
    "os.path module", "osgi alliance", "outer joins", "outriggers", "oversubscription",
    "paas (platform as a service)", "pack expansion", "packaged data models",
    "packaging service as zip file", "palm method", "parallel processing", "parallel quicksort",
    "parquet", "partial specialization", "partitioned hash joins", "path length", "peer review",
    "peer-to-peer pattern", "peer-to-peer systems", "perfect forwarding", "performance analysis",
    "periodic snapshot", "periodic snapshots", "persistence integration tests",
    "persistent storage", "persisting aggregates using events", "personas", "php api methods",
    "ping/echo", "pipe-and-filter pattern", "pipelining", "pkce flow",
    "platform as a service (paas)", "point-of-sale systems", "port forwarding", "port:",
    "portability", "pos (part-of-speech) tagging", "positional-only arguments", "pre-web era",
    "prefetching", "presentation area", "presumed abort", "pretraining", "preventing lost updates",
    "preventing write skew", "problem breakdown", "problems with hash mod n", "process group",
    "process pauses", "processor context", "producer properties", "producers", "product dimension",
    "production metrics", "progress graphs", "project structure", "property graphs",
    "protocol (amqp)", "protocol selection", "prototypes", "prototyping", "pseudo-conversations",
    "psql shell", "publicizing post-mortems", "publish-subscribe", "publish-subscribe pattern",
    "qa engineers", "quadtrees", "quality screens", "query optimization", "query optimizer",
    "query processing", "queryable state", "race conditions", "range partitioning", "rate limits",
    "read committed isolation", "read locks", "read quorum", "read uncommitted",
    "read-only transactions", "readability", "real-time analytics",
    "real-time data processing and analytics", "real-time processing", "real-time processing services",
    "real-time storage", "real-time systems", "reaping", "record stream", "recovery manager",
    "recursive", "recursive dimensional clustering (rdc)", "recursive query support",
    "recursive sql queries", "referential integrity", "register files", "registering classes",
    "regression tests", "relational algebra and sql", "relfrozenxid of table",
    "remote procedure call", "reordering imports", "replacement policies",
    "replication controller creation", "replication controller deletion",
    "representational state transfer (rest)", "reprocessing data", "request routing",
    "resilience engineering", "resource management", "rest (representational state transfer)",
    "retail use case", "return statement", "reverse proxy service", "risc design", "risc processors",
    "role playing", "rolling back", "rolling upgrades", "root exceptions for apis", "rpc systems",
    "ruby on rails", "rule-based", "run-time stack", "running as services", "running containers",
    "running default test suite", "running parametrized", "savepoints", "scalability (hotspots)",
    "scalability (number of open tasks)", "scaling out", "scaling up",
    "scds (slowly changing dimensions)", "scenario line", "scheduled services", "schema evolution",
    "search engine", "security vulnerability", "self-healing", "self-service metrics",
    "self-service platforms", "semantic functions", "semantic memory", "semi-additive facts",
    "semijoin operation", "sequential processing", "sequentially consistent ordering",
    "serializability", "serializable isolation", "server recovery", "service health",
    "service level performance", "service meshes", "service oriented architecture (soa)",
    "service platform selection", "service proxy", "service-oriented architecture",
    "service-oriented architecture pattern", "set associative caches", "set elements",
    "shadow maps", "shadow paging", "shared nothing", "shared state", "shared-data pattern",
    "shift operations", "short-circuiting", "shrunken rollup dimensions", "side effects",
    "silverlight", "simian army", "simulations", "single customer dimension",
    "single responsibility principle", "single-threaded execution", "skill keywords",
    "sloppy quorums", "sloppy quorums and hinted handoff", "smart keys", "smoke testing",
    "snapshot isolation support", "snowflaking", "socket function", "software raid",
    "source code integrity", "source code repository", "specifying prerequisites with givens",
    "specifying rest apis", "split and convert video lambda function", "spring framework",
    "sprints", "sql server service broker", "stable storage", "stack pointers",
    "staging environments", "star schemas", "starting with thens", "state changes",
    "state machine replication", "state machines", "state management", "stateless servers",
    "statement-based replication", "static libraries", "statistical analysis",
    "statistical and numerical algorithms", "std::add_lvalue_reference",
    "std::execution::sequenced_policy", "std::remove_const", "std::remove_reference",
    "std::vector", "step dimension", "stopiteration exception", "stored procedures",
    "str.format method", "stream joins", "streaming data ingestion", "streams and event processing",
    "striding", "stripe", "struct module", "structured logging", "stubs", "submissions queue",
    "subprocess module", "subsystems", "subtransactions", "subtype", "subtypes",
    "subword information", "subword models", "successor", "summary statistics", "super-peer",
    "surrogate keys", "swap space", "symbol tables", "system architecture", "system calls",
    "system management", "systems of record", "table creation", "table-testing criteria",
    "technical heterogeneity", "technical metadata layer", "term frequency (tf)",
    "termination protocol", "test types", "test-driven development", "testing environments",
    "testing in", "tests for order service", "text mining", "text parsers", "text string",
    "textual facts", "tf-idf calculation", "third-party", "thomas' write rule", "threading module",
    "threat landscape", "three-tier architecture", "three-tier web service",
    "thrift and protocol buffers", "ticket aggregate", "time module", "time points", "time-of-day",
    "timeout period", "timestamp ordering", "tokenization", "tombstone", "train-serve skew",
    "training model", "training pipeline", "transaction counts", "transaction handshakes",
    "transaction management", "transaction numbers", "transaction processing",
    "transactional messaging", "transactional rpc", "transcode video lambda function",
    "transcoding video", "transformer as language model", "transformer-xl",
    "transport layer security (tls)", "tuple type", "tuxedo", "tuxedo (oracle)", "two-phase",
    "two-phase commit", "two-phase commit (2pc)", "two-phase locking (2pl)", "two-pizza team",
    "type 1 in same dimension", "type 2 in same dimension", "typedef", "types of windows",
    "ubuntu instance", "unions", "upcasting", "upsampling and downsampling", "usability testing",
    "use of adapter", "use of bridge", "use of chain of responsibility", "use of command",
    "use of composite", "use of facade", "use of factory method", "use of interpreter",
    "use of iterator", "use of memento", "use of observer", "use of zookeeper", "user input",
    "user-defined", "using - -stateful=links", "using dapr components",
    "using mocks to verify behavior", "using one when per scenario",
    "utc (coordinated universal time)", "validating payloads with unknown fields",
    "validating request payloads with pydantic", "validating subclasses",
    "validating url query parameters", "value category", "variability", "vector similarity search",
    "version control systems", "version vectors", "virtual environments", "virtual machines (vms)",
    "virtual offices", "vms (virtual machines)", "vocabulary and token indexers", "void*",
    "warm backup", "warning notes", "weighting", "weighting losses", "wer (windows error reporting)",
    "white-box testing", "whitespace", "with pydantic", "with rdbms-based event store",
    "word associations", "work in progress (wip)", "work stealing", "write quorum", "write skew",
    "write skew (transaction isolation)", "ws-transactions", "xa interface", "xa transactions",
    "xml-rpc", "xpath", "y86-64 pipelining", "zero initialization", "zero-copy interactions",
    "zero-downtime deployments", "zero-shot cot prompting", "fine-grained", "first web era",
    "five-stage pipelines", "flash memory"
]

def main():
    # Load the validated filter
    filter_path = Path('/Users/kevintoles/POC/Code-Orchestrator-Service/data/validated_term_filter.json')
    with open(filter_path) as f:
        data = json.load(f)

    all_filter = set(k.lower() for k in data.get('keywords', []) + data.get('concepts', []))
    
    print(f"Total KEEP terms to verify: {len(KEEP_TERMS)}")
    print(f"Filter contains: {len(all_filter)} unique terms")
    print()
    
    # Check filter
    in_filter = [t for t in KEEP_TERMS if t.lower() in all_filter]
    not_in_filter = [t for t in KEEP_TERMS if t.lower() not in all_filter]
    
    print(f"=== FILTER CHECK ===")
    print(f"IN validated_term_filter.json: {len(in_filter)}/{len(KEEP_TERMS)} ({100*len(in_filter)/len(KEEP_TERMS):.1f}%)")
    print(f"NOT in filter: {len(not_in_filter)}")
    
    if not_in_filter:
        print("\nMissing from filter:")
        for t in not_in_filter:
            print(f"  ❌ {t}")
    
    # Check chapters
    print(f"\n=== CHAPTER BACKFILL CHECK ===")
    metadata_dir = Path('/Users/kevintoles/POC/llm-document-enhancer/workflows/metadata_extraction/output')
    
    keep_lower = set(t.lower() for t in KEEP_TERMS)
    found_in_chapters = set()
    
    for f in metadata_dir.glob('*.json'):
        try:
            chapters = json.load(open(f))
            if isinstance(chapters, list):
                for ch in chapters:
                    for kw in ch.get('keywords', []):
                        if kw.lower() in keep_lower:
                            found_in_chapters.add(kw.lower())
        except:
            pass
    
    print(f"KEEP terms in chapter metadata: {len(found_in_chapters)}/{len(KEEP_TERMS)} ({100*len(found_in_chapters)/len(KEEP_TERMS):.1f}%)")
    
    not_backfilled = [t for t in KEEP_TERMS if t.lower() not in found_in_chapters]
    print(f"NOT in any chapter: {len(not_backfilled)}")
    
    if not_backfilled and len(not_backfilled) <= 60:
        print("\nNot backfilled to chapters:")
        for t in not_backfilled:
            print(f"  ⚠️  {t}")


if __name__ == '__main__':
    main()
