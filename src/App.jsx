import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  ArrowDown, ArrowRight, ArrowUpRight, BookOpen, Check,
  CircleHelp, Clock3, Copy, Cpu, Database, FileText,
  Layers3, Menu, MessageSquareText, Play, Search, Sparkles, X,
} from "lucide-react";

const sections = [
  ["01", "Motivation", "motivation"],
  ["02", "What is RAG?", "what-is-rag"],
  ["03", "Chunking", "chunking"],
  ["04", "Embeddings", "embeddings"],
  ["05", "Cosine similarity", "similarity"],
  ["06", "Generation", "generation"],
  ["07", "Summary", "summary"],
];

const exampleDocument = `Retrieval-augmented generation (RAG) is a way to give a language model useful information at the moment it answers a question. A model's training data does not automatically include a team's latest policies, product documentation, or private knowledge base. RAG adds a retrieval step so the application can find relevant information and provide it as context.

The process begins with source documents such as handbooks, support articles, research papers, and product guides. Documents are cleaned and divided into smaller sections called chunks. Chunking matters because a whole book is too large and unfocused to pass to a model for every question. A good chunk is large enough to preserve an idea, but small enough that search can identify the useful passage. Some systems split by a fixed number of words or tokens. Others split at sentence, heading, or semantic boundaries. Overlap repeats a small amount of text across neighboring chunks so a definition or sentence is less likely to be cut off at a boundary.

Next, an embedding model converts each chunk into a vector: an ordered list of numbers that represents aspects of its meaning. The original chunk text and its vector are stored together in a vector database. A vector database indexes those vectors so it can find nearby items efficiently, even when a collection contains many documents. Metadata such as document title, section, date, or access permissions can be stored with each chunk as well.

When a participant asks a question, the same embedding model converts that question into a query vector. The retrieval step compares the query vector with stored chunk vectors. Cosine similarity is one common measure: it compares the angle between two vectors, with a smaller angle indicating more similar directions. The database returns the chunks with the strongest scores. This can find a passage about “resetting a password” for a question phrased as “I cannot sign in,” even though the wording is different.

The application then builds a prompt containing instructions, the participant's question, and the selected passages. The generation model uses that context to compose an answer. Instructions can ask it to rely on the supplied material, cite relevant sections, and say when the documents do not contain enough information. Retrieval does not guarantee a correct response, so teams should evaluate chunk quality, search results, and generated answers with realistic questions.

This workshop follows that same sequence. Adjust chunk size and overlap, compare each chunk with a search question, inspect a small embedding vector, and compare sentence similarity. Then connect the pieces: split source material, embed the chunks, retrieve the most relevant passages, and give those passages to a model as context. The goal is to build an intuition for how the parts work together, not to memorize a list of terms.`;

const chunkMethods = [
  { id: "fixed_size", number: "01", title: "Fixed-size", description: "Same-size windows" },
  { id: "sentence", number: "02", title: "Sentence", description: "Keep sentences whole" },
  { id: "semantic", number: "03", title: "Semantic", description: "Split when meaning shifts" },
];

function previewChunks(text, size, overlap, method) {
  const sentences = text.trim().split(/(?<=[.!?])\s+|\n+/).filter(Boolean).flatMap((sentence) => {
    const words = sentence.split(/\s+/);
    return Array.from({ length: Math.ceil(words.length / size) }, (_, index) =>
      words.slice(index * size, (index + 1) * size).join(" "),
    );
  });
  if (method === "fixed_size") {
    const words = text.trim().split(/\s+/).filter(Boolean);
    const step = Math.max(1, size - overlap);
    const chunks = [];
    for (let start = 0; start < words.length; start += step) {
      chunks.push(words.slice(start, start + size).join(" "));
      if (start + size >= words.length) break;
    }
    return chunks;
  }

  const groups = [];
  let current = [];
  let currentWords = 0;
  for (const sentence of sentences) {
    const sentenceWords = sentence.split(/\s+/);
    if (current.length && currentWords + sentenceWords.length > size) {
      groups.push(current);
      current = [];
      currentWords = 0;
    }
    current.push(sentence);
    currentWords += sentenceWords.length;
  }
  if (current.length) groups.push(current);

  return groups.map((group, index) => {
    const previousWords = index > 0 ? groups[index - 1].join(" ").split(/\s+/) : [];
    const prefix = overlap ? previousWords.slice(-overlap) : [];
    return [...prefix, ...group.join(" ").split(/\s+/)].join(" ");
  });
}

const fadeUp = {
  hidden: { opacity: 0, y: 18 },
  show: { opacity: 1, y: 0, transition: { duration: 0.55, ease: [0.22, 1, 0.36, 1] } },
};

function VectorTitle({ title, as: Tag = "h2", className = "" }) {
  const ref = useRef(null);
  const [progress, setProgress] = useState(0);
  const started = progress > 0;
  const revealed = progress >= title.length;
  const vectorGlyphs = "0123456789.,-+[]{}<>";
  const displayText = [...title].map((character, index) => {
    if (/\s/.test(character) || index < progress) return character;
    return vectorGlyphs[(index * 7 + progress * 11) % vectorGlyphs.length];
  }).join("");

  useEffect(() => {
    const element = ref.current;
    if (!element || revealed) return undefined;

    const revealOnScroll = () => {
      const bounds = element.getBoundingClientRect();
      if (bounds.top >= window.innerHeight * 0.9 || bounds.bottom <= 0) return;
      window.removeEventListener("scroll", revealOnScroll);
      window.removeEventListener("resize", revealOnScroll);
      if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
        setProgress(title.length);
        return;
      }

      let nextProgress = 0;
      timer = window.setInterval(() => {
        nextProgress = Math.min(title.length, nextProgress + Math.max(1, Math.ceil(title.length / 26)));
        setProgress(nextProgress);
        if (nextProgress >= title.length) window.clearInterval(timer);
      }, 20);
    };

    let timer;
    window.addEventListener("scroll", revealOnScroll, { passive: true });
    window.addEventListener("resize", revealOnScroll);
    revealOnScroll();
    return () => {
      window.removeEventListener("scroll", revealOnScroll);
      window.removeEventListener("resize", revealOnScroll);
      window.clearInterval(timer);
    };
  }, [revealed, title]);

  return <Tag ref={ref} className={`vector-title ${className}`.trim()} aria-label={title}>
    <span aria-hidden="true" className="vector-title-layout">{title}</span>
    <span aria-hidden="true" className={started ? "vector-title-vector is-hidden" : "vector-title-vector"}>
      [ 0.42, −0.18, 0.76, 0.31, −0.54, 0.09, 0.63, −0.27, … ]
    </span>
    <span aria-hidden="true" className={`vector-title-text${started ? " is-started" : ""}${revealed ? " is-revealed" : ""}`}>
      {displayText}
    </span>
  </Tag>;
}

function SectionHeading({ number, title, children }) {
  return (
    <motion.div className="section-heading" variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true, amount: 0.2 }}>
      <span className="section-number">{number}</span>
      <div><VectorTitle title={title} />{children && <p className="section-intro">{children}</p>}</div>
    </motion.div>
  );
}

function AnimatedPipeline() {
  const steps = [
    { icon: FileText, title: "Your docs", color: "lilac" },
    { icon: Layers3, title: "Chunks", color: "peach" },
    { icon: Cpu, title: "Vectors", color: "mint" },
    { icon: Database, title: "Vector DB", color: "blue" },
    { icon: Sparkles, title: "Good answers", color: "yellow" },
  ];
  return <div className="pipeline" aria-label="Documents become chunks, vectors, search results, and grounded answers">
    {steps.map(({ icon: Icon, title, color }, i) => <div className="pipeline-step" key={title}>
      <motion.div className={`pipeline-icon ${color}`} animate={{ y: [0, -5, 0] }} transition={{ duration: 2.5, delay: i * 0.25, repeat: Infinity, ease: "easeInOut" }}><Icon size={21} strokeWidth={1.8} /></motion.div>
      <span>{title}</span>
      {i < steps.length - 1 && <div className="pipeline-connector"><motion.i animate={{ scaleX: [0, 1, 0], opacity: [0.25, 1, 0.25] }} transition={{ duration: 1.8, delay: i * 0.25, repeat: Infinity }} /><ArrowRight size={14} /></div>}
    </div>)}
  </div>;
}

function ChunkVisualizer({ chunks, overlap, scores, scoresLoading }) {
  const [visibleCount, setVisibleCount] = useState(8);
  useEffect(() => {
    setVisibleCount(8);
  }, [chunks, overlap]);

  return <div className="chunk-visualizer">
    <div className="chunk-legend"><span><i className="legend-dot" /> chunk</span><span><i className="legend-dot overlap-dot" /> overlap</span></div>
    <div className="chunk-cards">
      {chunks.slice(0, visibleCount).map((chunk, index) => {
        const chunkWords = chunk.split(/\s+/).filter(Boolean);
        const previousWords = index > 0 ? chunks[index - 1].split(/\s+/).filter(Boolean) : [];
        let overlapPrefix = 0;
        for (let length = Math.min(overlap, chunkWords.length, previousWords.length); length > 0; length -= 1) {
          const prefix = chunkWords.slice(0, length).map((word) => word.toLowerCase());
          const suffix = previousWords.slice(-length).map((word) => word.toLowerCase());
          if (prefix.every((word, wordIndex) => word === suffix[wordIndex])) {
            overlapPrefix = length;
            break;
          }
        }
        return <motion.details className="chunk-card" key={`${index}-${chunkWords[0]}`} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: index * 0.06 }} open={index === 0}>
        <summary className="chunk-card-summary">
          <div className="chunk-card-head"><span>CHUNK {String(index + 1).padStart(2, "0")}</span><span>{chunkWords.length} words</span></div>
          <div className="chunk-score-row"><span>SIMILARITY TO QUERY</span><b>{scoresLoading ? "…" : Number.isFinite(scores[index]) ? scores[index].toFixed(2) : "—"}</b></div>
          <div className="chunk-score-meter"><i style={{ width: Number.isFinite(scores[index]) ? `${Math.max(0, Math.min(100, (scores[index] + 1) * 50))}%` : "0%" }} /></div>
        </summary>
        <p>{chunkWords.map((word, wordIndex) => {
          return <span className={wordIndex < overlapPrefix ? "word-overlap" : ""} key={`${wordIndex}-${word}`}>{word} </span>;
        })}</p>
      </motion.details>;
      })}
      {chunks.length > visibleCount && <button className="more-chunks" type="button" onClick={() => setVisibleCount((count) => Math.min(count + 8, chunks.length))}>
        Show the next {Math.min(8, chunks.length - visibleCount)} chunks ({chunks.length - visibleCount} remaining)
      </button>}
    </div>
  </div>;
}

function App() {
  const [activeSection, setActiveSection] = useState("motivation");
  const [mobileMenu, setMobileMenu] = useState(false);
  const [documentText, setDocumentText] = useState(exampleDocument);
  const [chunkSize, setChunkSize] = useState(48);
  const [overlap, setOverlap] = useState(10);
  const [chunkMethod, setChunkMethod] = useState("fixed_size");
  const [chunkQuery, setChunkQuery] = useState("What is RAG?");
  const [chunkScores, setChunkScores] = useState([]);
  const [chunkResult, setChunkResult] = useState(null);
  const [chunkScoresLoading, setChunkScoresLoading] = useState(false);
  const [chunkScoresError, setChunkScoresError] = useState("");
  const [sentence, setSentence] = useState("The cat sat on the windowsill.");
  const [embedding, setEmbedding] = useState(null);
  const [showFullEmbedding, setShowFullEmbedding] = useState(false);
  const [embeddingLoading, setEmbeddingLoading] = useState(false);
  const [embeddingError, setEmbeddingError] = useState("");
  const [sentenceA, setSentenceA] = useState("A dog is playing in the park.");
  const [sentenceB, setSentenceB] = useState("A puppy runs around outside.");
  const [similarity, setSimilarity] = useState(null);
  const [similarityLoading, setSimilarityLoading] = useState(false);
  const [similarityError, setSimilarityError] = useState("");
  const [systemPrompt, setSystemPrompt] = useState("You are a helpful assistant. Answer using only the context provided. If the answer is not in the context, say you don't know.");
  const [userPrompt, setUserPrompt] = useState("What are the steps in a RAG pipeline?");
  const [answer, setAnswer] = useState("");
  const [retrievedChunks, setRetrievedChunks] = useState([]);
  const [answerLoading, setAnswerLoading] = useState(false);
  const [answerError, setAnswerError] = useState("");
  const [copied, setCopied] = useState(false);

  const words = useMemo(() => documentText.trim().split(/\s+/).filter(Boolean), [documentText]);
  const safeOverlap = Math.min(overlap, chunkSize - 1);
  const localChunkPreview = useMemo(
    () => previewChunks(documentText, chunkSize, safeOverlap, chunkMethod),
    [documentText, chunkSize, safeOverlap, chunkMethod],
  );
  const displayedChunks = chunkResult?.chunks ?? localChunkPreview;
  const estimatedTime = Math.max(0.2, (words.length * 0.012 + displayedChunks.length * 0.04)).toFixed(1);
  const [chunkTiming, setChunkTiming] = useState(null);

  useEffect(() => {
    const controller = new AbortController();
    setChunkTiming(null);
    setChunkResult(null);
    setChunkScores([]);
    setChunkScoresError("");
    setChunkScoresLoading(Boolean(chunkQuery.trim()));
    const timer = window.setTimeout(async () => {
      try {
        const response = await fetch("/api/chunk", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: documentText, chunk_size: chunkSize, overlap: safeOverlap, query: chunkQuery.trim() || undefined, method: chunkMethod }),
          signal: controller.signal,
        });
        let data;
        try {
          data = await response.json();
        } catch {
          throw new Error(`API returned an invalid response (${response.status}). Check that the Python backend is running.`);
        }
        if (!response.ok) throw new Error(data.detail || "Chunking API is unavailable.");
        setChunkTiming({ milliseconds: data.processing_ms, count: data.count });
        setChunkResult(data);
        setChunkScores(data.similarities || []);
      } catch (error) {
        if (error.name !== "AbortError") {
          setChunkTiming(null);
          setChunkScoresError(error.message || "Could not calculate chunk similarity scores.");
        }
      } finally {
        if (!controller.signal.aborted) setChunkScoresLoading(false);
      }
    }, 250);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [documentText, chunkSize, safeOverlap, chunkQuery, chunkMethod]);

  const displayedChunkCount = chunkTiming?.count ?? localChunkPreview.length;

  const scrollToSection = (id) => {
    setActiveSection(id);
    setMobileMenu(false);
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  async function callApi(path, payload) {
    const response = await fetch(`/api/${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error(`API returned an invalid response (${response.status}).`);
    }
    if (!response.ok) throw new Error(data.detail || "Something went wrong. Please try again.");
    return data;
  }

  async function generateEmbedding() {
    setEmbeddingLoading(true); setEmbeddingError(""); setEmbedding(null); setShowFullEmbedding(false);
    try { setEmbedding(await callApi("embed", { text: sentence })); }
    catch (error) { setEmbeddingError(error.message); }
    finally { setEmbeddingLoading(false); }
  }

  async function compareSentences() {
    setSimilarityLoading(true); setSimilarityError(""); setSimilarity(null);
    try { setSimilarity(await callApi("similarity", { text_a: sentenceA, text_b: sentenceB })); }
    catch (error) { setSimilarityError(error.message); }
    finally { setSimilarityLoading(false); }
  }

  async function runGeneration() {
    setAnswerLoading(true); setAnswerError(""); setAnswer(""); setRetrievedChunks([]);
    try {
      const data = await callApi("generate", { system_prompt: systemPrompt, user_prompt: userPrompt, context: documentText, chunk_size: chunkSize, overlap: safeOverlap, chunk_method: chunkMethod });
      setAnswer(data.answer);
      setRetrievedChunks(data.retrieved_chunks || []);
    } catch (error) { setAnswerError(error.message); }
    finally { setAnswerLoading(false); }
  }

  async function copyAnswer() {
    await navigator.clipboard.writeText(answer);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  }

  return <div className="app-shell">
    <header className="topbar">
      <a className="brand" href="#top" onClick={() => setActiveSection("motivation")}><img className="brand-mark" src="/delta-mark.svg" alt="" /><span className="brand-name">DELTA <span>WORKSHOPS</span></span></a>
      <button className="mobile-menu-toggle" aria-label="Toggle navigation" onClick={() => setMobileMenu(!mobileMenu)}>{mobileMenu ? <X /> : <Menu />}</button>
      <nav className={mobileMenu ? "top-nav nav-open" : "top-nav"} aria-label="Tutorial sections">
        {sections.map(([number, label, id]) => <button className={activeSection === id ? "nav-link active" : "nav-link"} key={id} onClick={() => scrollToSection(id)}><span>{number}</span>{label}</button>)}
      </nav>
      <a className="top-cta" href="#chunking" onClick={() => setActiveSection("chunking")}>Jump into the lab <ArrowUpRight size={14} /></a>
    </header>

    <main id="top">
      <section className="hero content-width" id="motivation">
        <motion.div className="hero-copy" initial="hidden" animate="show" variants={{ show: { transition: { staggerChildren: 0.12 } } }}>
          <motion.div variants={fadeUp}><VectorTitle as="h1" title="A practical introduction to retrieval-augmented generation" /></motion.div>
          <motion.p variants={fadeUp}>Learn how RAG retrieves relevant information and uses it to generate grounded answers. Explore chunking, embeddings, similarity, and prompting.</motion.p>
          <motion.div variants={fadeUp} className="hero-actions"><button className="button button-dark" onClick={() => scrollToSection("what-is-rag")}>Start exploring <ArrowDown size={16} /></button></motion.div>
        </motion.div>
        <motion.div className="hero-art" initial={{ opacity: 0, scale: 0.94 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.8 }}>
          <div className="art-glow" />
          <AnimatedPipeline />
        </motion.div>
      </section>

      <section className="motivation-band">
        <div className="content-width motivation-inner">
          <SectionHeading number="01" title="Why use RAG?">
            Language models can give incorrect or outdated answers because their training data is fixed. RAG retrieves relevant information from your documents before the model responds.
          </SectionHeading>
        </div>
      </section>

      <section className="intro-section content-width" id="what-is-rag">
        <SectionHeading number="02" title="What is RAG?">
          Retrieval-augmented generation gives a model a cheat sheet. First retrieve relevant information, then generate an answer grounded in it.
        </SectionHeading>
        <div className="rag-explainer">
          <div className="rag-definition"><div className="definition-icon"><BookOpen size={20} /></div><p><b>RAG = Retrieve + Augment + Generate.</b> The system fetches relevant information when a question is asked, then uses it to help generate the answer.</p></div>
          <div className="why-cards"><article><span className="card-icon lilac"><Search size={17} /></span><h3>Up-to-date</h3><p>Bring your own docs. No retraining required.</p></article><article><span className="card-icon mint"><Check size={17} /></span><h3>More grounded</h3><p>Give answers a source they can point to.</p></article><article><span className="card-icon peach"><Sparkles size={17} /></span><h3>More useful</h3><p>Let the model work with your knowledge.</p></article></div>
        </div>
      </section>

      <section className="chunk-section" id="chunking">
        <div className="content-width chunk-layout">
          <div className="chunk-copy"><SectionHeading number="03" title="How document chunking works">
            Models can’t retrieve a whole library at once. Chunking slices your documents into bite-sized passages that are easy to find and fit in a prompt.
          </SectionHeading><div className="mini-tip"><span>✳</span><p><b>Why overlap?</b> A little shared text helps keep ideas intact when a sentence lands on a chunk boundary.</p></div>
          <div className="chunk-methods"><p>THREE WAYS TO SLICE IT</p><div role="group" aria-label="Choose a chunking method">{chunkMethods.map((method) => <button key={method.id} type="button" className={chunkMethod === method.id ? "method-active" : ""} aria-pressed={chunkMethod === method.id} onClick={() => setChunkMethod(method.id)}><b>{method.number}</b><strong>{method.title}</strong><small>{method.description}</small></button>)}</div><p className="chunk-method-note">{chunkMethod === "semantic" ? "Semantic mode embeds each sentence and starts a new chunk where adjacent meaning shifts." : chunkMethod === "sentence" ? "Sentence mode keeps sentence boundaries intact and groups sentences up to the selected size." : "Fixed-size mode creates evenly sized word windows using the selected overlap."}</p></div></div>
          <div className="lab-card chunk-lab">
            <div className="lab-header"><div><span className="lab-eyebrow">INTERACTIVE LAB 01</span><VectorTitle as="h3" title="Chunking playground" /></div><span className="lab-status"><i /> LIVE</span></div>
            <label className="field-label" htmlFor="document-text">YOUR SOURCE DOCUMENT <span>{words.length} words</span></label>
            <textarea id="document-text" className="document-input" value={documentText} onChange={(event) => setDocumentText(event.target.value)} maxLength={10000} />
            <label className="field-label" htmlFor="chunk-query">QUESTION TO COMPARE WITH EACH CHUNK</label>
            <input id="chunk-query" className="similarity-input chunk-query-input" value={chunkQuery} maxLength={200} onChange={(event) => setChunkQuery(event.target.value)} placeholder="Try: How does retrieval find relevant passages?" />
            {chunkScoresError && <div className="error-message" role="alert">{chunkScoresError}</div>}
            <div className="slider-group"><div className="slider-label"><label htmlFor="chunk-size">Chunk size</label><span><b>{chunkSize}</b> words</span></div><input id="chunk-size" type="range" min="12" max="120" value={chunkSize} onChange={(event) => { const next = Number(event.target.value); setChunkSize(next); setOverlap((value) => Math.min(value, next - 1)); }} /><div className="range-ends"><span>12 words</span><span>120 words</span></div></div>
            <div className="slider-group"><div className="slider-label"><label htmlFor="chunk-overlap">Chunk overlap</label><span><b>{safeOverlap}</b> words</span></div><input id="chunk-overlap" type="range" min="0" max={Math.max(1, Math.min(40, chunkSize - 1))} value={safeOverlap} onChange={(event) => setOverlap(Number(event.target.value))} /><div className="range-ends"><span>None</span><span>More context</span></div></div>
            <div className="stats-row"><div><span>CHUNKS</span><b>{displayedChunkCount}</b></div><div><span>PROCESSING TIME <CircleHelp size={12} /></span><b>{chunkTiming ? (chunkTiming.milliseconds < 10 ? "< 10" : chunkTiming.milliseconds.toFixed(1)) : `~${estimatedTime}`}<small>{chunkTiming ? " ms" : " sec"}</small></b></div><div><span>CHARS</span><b>{documentText.length.toLocaleString()}</b></div></div>
            <ChunkVisualizer chunks={displayedChunks} overlap={safeOverlap} scores={chunkScores} scoresLoading={chunkScoresLoading} />
            <p className="chunk-score-caption">Cosine similarity ranges from −1 to 1. Higher scores indicate chunks more related to the question.</p>
            <p className="lab-footnote"><Clock3 size={13} /> {chunkTiming ? "Measured server-side split time. The API response also confirms your chunk count." : "Estimated split time shown while the API is offline."}</p>
          </div>
        </div>
      </section>

      <section className="embedding-section content-width" id="embeddings">
        <div className="embedding-copy"><SectionHeading number="04" title="How embeddings represent text">
          An embedding model turns text into an ordered list of numbers called a vector. Similar ideas land near each other—even if they use different words.
        </SectionHeading><div className="embedding-model-note"><div className="model-icon"><Cpu size={17} /></div><div><span>THE LITTLE MODEL</span><b>all-MiniLM-L6-v2</b><small>384 dimensions · runs via Hugging Face</small></div><span className="model-badge">SMALL + FAST</span></div></div>
        <div className="lab-card embed-lab"><div className="lab-header"><div><span className="lab-eyebrow">INTERACTIVE LAB 02</span><VectorTitle as="h3" title="Peek inside an embedding" /></div><span className="model-tag"><Cpu size={13} /> 384-D</span></div>
          <label className="field-label" htmlFor="embed-sentence">TYPE A SENTENCE <span>{sentence.length}/200</span></label><textarea id="embed-sentence" value={sentence} onChange={(event) => setSentence(event.target.value.slice(0, 200))} maxLength={200} placeholder="Try: A cat naps in the sun." className="sentence-input" />
          <button className="button button-primary full-button" onClick={generateEmbedding} disabled={!sentence.trim() || embeddingLoading}><Play size={14} fill="currentColor" />{embeddingLoading ? "Creating the vector…" : "Generate embedding"}<ArrowRight size={15} /></button>
          {embeddingError && <div className="error-message" role="alert">{embeddingError}</div>}
          {embedding ? <div className="vector-output">
            <div className="vector-output-head"><span>YOUR VECTOR <b>{embedding.dimensions} values</b></span><span className="success-dot">● MODEL READY</span></div>
            <pre className="vector-array">[{showFullEmbedding ? `\n  ${embedding.vector.map((value) => value.toFixed(6)).join(",\n  ")}\n` : `${embedding.vector.slice(0, 10).map((value) => value.toFixed(6)).join(", ")}, …`}]</pre>
            <button className="vector-array-toggle" type="button" onClick={() => setShowFullEmbedding((visible) => !visible)} aria-expanded={showFullEmbedding}>
              {showFullEmbedding ? "Hide full array" : `Click to view all ${embedding.dimensions} values`}
            </button>
            <div className="vector-note">{showFullEmbedding ? "All vector dimensions are shown in order." : `Showing 10 of ${embedding.dimensions} values. The array is ordered: each position is a vector dimension.`}</div>
          </div> : <div className="vector-placeholder"><code>[ 0.12, −0.04, 0.87, … ]</code><span>Your vector will appear here</span></div>}
        </div>
      </section>

      <section className="similarity-section" id="similarity"><div className="content-width similarity-layout">
        <div><SectionHeading number="05" title="How cosine similarity works">
          A vector database compares your question with stored chunks. Cosine similarity checks if two vectors point in a similar direction: closer angle, closer meaning.
        </SectionHeading><div className="cosine-visual">
          <svg className="cosine-diagram" viewBox="0 0 300 205" role="img" aria-labelledby="cosine-title cosine-description">
            <title id="cosine-title">Cosine similarity angle between two sentence vectors</title>
            <desc id="cosine-description">Two vectors start at the same origin. The green arc marks the angle between the solid vector A and dashed vector B.</desc>
            <defs>
              <marker id="axis-arrow" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 Z" /></marker>
              <marker id="vector-arrow-a" markerUnits="userSpaceOnUse" markerWidth="13" markerHeight="13" refX="11" refY="6.5" orient="auto"><path d="M0 0 L13 6.5 L0 13 Z" /></marker>
              <marker id="vector-arrow-b" markerUnits="userSpaceOnUse" markerWidth="13" markerHeight="13" refX="11" refY="6.5" orient="auto"><path d="M0 0 L13 6.5 L0 13 Z" /></marker>
            </defs>
            <line className="cosine-axis" x1="20" y1="168" x2="278" y2="168" markerEnd="url(#axis-arrow)" />
            <line className="cosine-axis" x1="34" y1="184" x2="34" y2="20" markerEnd="url(#axis-arrow)" />
            <line className="cosine-vector-a" x1="34" y1="168" x2="150" y2="52" markerEnd="url(#vector-arrow-a)" />
            <line className="cosine-vector-b" x1="34" y1="168" x2="235" y2="95" markerEnd="url(#vector-arrow-b)" />
            <circle className="cosine-origin" cx="34" cy="168" r="4" />
            <path className="cosine-angle-arc" d="M67.9 134.1 A48 48 0 0 1 79.1 151.6" />
            <rect className="cosine-angle-label-bg" x="43" y="136" width="38" height="20" rx="4" />
            <text className="cosine-theta" x="48" y="150">θ°</text>
            <text className="cosine-vector-label vector-label-a" x="111" y="43">A — solid</text>
            <text className="cosine-vector-label vector-label-b" x="175" y="80">B - dashed</text>
            <text className="cosine-axis-label" x="247" y="185">dimension 1</text>
            <text className="cosine-caption" x="5" y="13">angle between vectors</text>
          </svg>
        </div><div className="db-footnote"><Database size={15} /><span><b>Vector DBs</b> index these vectors so nearest-neighbor search stays fast—even across millions of chunks.</span></div></div>
        <div className="lab-card similarity-lab"><div className="lab-header"><div><span className="lab-eyebrow">INTERACTIVE LAB 03</span><VectorTitle as="h3" title="Meaning, meet math" /></div><span className="model-tag">MAX 100 CHARS</span></div>
          <label className="field-label" htmlFor="sentence-a">SENTENCE A <span>{sentenceA.length}/100</span></label><input id="sentence-a" className="similarity-input" value={sentenceA} maxLength={100} onChange={(event) => setSentenceA(event.target.value)} />
          <label className="field-label" htmlFor="sentence-b">SENTENCE B <span>{sentenceB.length}/100</span></label><input id="sentence-b" className="similarity-input" value={sentenceB} maxLength={100} onChange={(event) => setSentenceB(event.target.value)} />
          <button className="button button-dark full-button" onClick={compareSentences} disabled={!sentenceA.trim() || !sentenceB.trim() || similarityLoading}><Sparkles size={15} />{similarityLoading ? "Comparing meaning…" : "Compare sentences"}<ArrowRight size={15} /></button>
          {similarityError && <div className="error-message" role="alert">{similarityError}</div>}
          {similarity ? <div className="similarity-result"><div className="similarity-score"><span className="score-number">{similarity.similarity.toFixed(2)}</span><span className="score-label">COSINE SIMILARITY · −1 TO 1</span></div><div className="score-meter"><i style={{ width: `${Math.max(0, Math.min(100, similarity.similarity * 100))}%` }} /></div><p>{similarity.interpretation}</p></div> : <div className="similarity-placeholder"><span>↗</span><p>Change a sentence and compare. Related meanings score higher.</p></div>}
          <div className="score-key"><span><i className="key-dot high" /> closer meaning</span><span><i className="key-dot low" /> less related</span></div>
        </div>
      </div></section>

      <section className="generation-section content-width" id="generation">
        <div className="generation-copy"><SectionHeading number="06" title="Generate an answer from retrieved context">
          We put the retrieved chunks into a prompt. Clear instructions tell the model how to use that context—and when not to guess.
        </SectionHeading><div className="prompt-tip"><MessageSquareText size={16} /><p><b>Prompt engineering:</b> Give the model a role, relevant context, and a clear task. Ask it to say when the context does not provide the answer.</p></div></div>
        <div className="lab-card generation-lab"><div className="lab-header"><div><span className="lab-eyebrow">THE GENERATION STAGE</span><VectorTitle as="h3" title="Put it all together" /></div><span className="model-tag google-tag"><Sparkles size={13} /> GOOGLE GEMINI</span></div>
          <label className="field-label" htmlFor="system-prompt">SYSTEM PROMPT <span>THE RULES</span></label><textarea id="system-prompt" className="prompt-input system-input" value={systemPrompt} onChange={(event) => setSystemPrompt(event.target.value)} maxLength={2000} />
          <label className="field-label" htmlFor="user-prompt">USER PROMPT <span>THE QUESTION</span></label><textarea id="user-prompt" className="prompt-input user-input" value={userPrompt} onChange={(event) => setUserPrompt(event.target.value)} maxLength={1000} />
          <div className="context-preview"><div className="context-preview-head"><span><BookOpen size={13} /> {retrievedChunks.length ? "TOP MATCHING CHUNKS" : "READY TO RETRIEVE"}</span><span>{retrievedChunks.length || Math.min(3, displayedChunks.length)} chunks</span></div><p>{retrievedChunks.length ? retrievedChunks.join("\n\n") : `${documentText.trim().slice(0, 240)}${documentText.length > 240 ? "…" : ""}`}</p></div>
          <button className="button button-primary full-button" onClick={runGeneration} disabled={!systemPrompt.trim() || !userPrompt.trim() || answerLoading}><Sparkles size={15} />{answerLoading ? "Thinking with context…" : "Generate an answer"}<ArrowRight size={15} /></button>
          {answerError && <div className="error-message" role="alert">{answerError}</div>}
          {answer && <motion.div className="answer-output" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}><div className="answer-output-head"><span><Sparkles size={14} /> GROUNDED ANSWER</span><button aria-label="Copy answer" onClick={copyAnswer}>{copied ? <Check size={14} /> : <Copy size={14} />}{copied ? "Copied" : "Copy"}</button></div><p>{answer}</p></motion.div>}
          {!answer && !answerError && <div className="answer-hint"><span>✳</span> Your answer appears here, anchored to your document.</div>}
        </div>
      </section>

      <section className="summary-section" id="summary"><div className="content-width summary-content"><SectionHeading number="07" title="RAG process summary">
        </SectionHeading><div className="recap-flow">{[["01", "Split", "Chunk your docs into useful pieces."], ["02", "Map", "Embed every chunk into a vector."], ["03", "Store", "Save vectors in a searchable database."], ["04", "Retrieve", "Find chunks close to the question."], ["05", "Generate", "Give the model context. Get a grounded answer."]].map(([num, title, desc], index) => <motion.div className="recap-item" key={title} initial={{ opacity: 0, y: 12 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: index * 0.08 }}><span className="recap-number">{num}</span><div><b>{title}</b><span>{desc}</span></div>{index < 4 && <ArrowRight className="recap-arrow" size={16} />}</motion.div>)}</div><div className="summary-footer"><span>And that’s retrieval-augmented generation.</span><button onClick={() => scrollToSection("motivation")}>Back to the beginning <ArrowUpRight size={15} /></button></div></div></section>
    </main>
    <footer className="footer"><div className="content-width footer-inner"><a className="brand" href="#top"><img className="brand-mark" src="/delta-mark.svg" alt="" /><span className="brand-name">DELTA <span>WORKSHOPS</span></span></a><span>© 2026 Delta Force. All Rights Reserved.</span><span className="footer-credit">Made with <span aria-label="love">♥</span> by Delta Force</span></div></footer>
  </div>;
}

export default App;
