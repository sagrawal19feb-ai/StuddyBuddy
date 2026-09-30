"""
search_engine.py — Web search & resource discovery for StudyBuddy
===================================================================

StudyBuddy can search the live web and return study-friendly results.
Requests like *"search Python loops"*, *"find notes for Biology"* or
*"need Java tutorial"* are turned into clean search queries, executed
against DuckDuckGo's HTML endpoint (no API key required), parsed with
BeautifulSoup, and ranked so that trusted educational domains
(``geeksforgeeks.org``, ``w3schools.com``, ``python.org``, ``ncert.nic.in``,
``khanacademy.org``, ``wikipedia.org``, ``youtube.com``, ``github.com``,
``scholar.google.com``) rise to the top.

The engine is fully defensive:

  * network failures fall back to curated offline links from the knowledge
    base with a friendly note,
  * the ``webbrowser`` module opens the user's default browser on demand,
  * every query is logged and remembered for "recent searches".

Beyond listing results, StudyBuddy can **extract an answer from the web**:
:meth:`SearchEngine.extract_answer` fetches the top result pages, strips
navigation and scripts, scores the paragraphs against the query keywords,
and returns the single best passage — so out-of-knowledge questions get a
real answer with its source, not just a list of links.
"""

from __future__ import annotations

import html as html_lib
import re
import webbrowser
from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup

from config import RUNTIME
from knowledge import KnowledgeBase
from logger import get_logger

log = get_logger("search_engine")

# Domains that are great for studying, with a ranking bonus.
PRIORITY_DOMAINS: dict[str, int] = {
    "geeksforgeeks.org": 6,
    "w3schools.com": 6,
    "python.org": 6,
    "khanacademy.org": 6,
    "ncert.nic.in": 5,
    "wikipedia.org": 4,
    "youtube.com": 4,
    "github.com": 3,
    "scholar.google.com": 5,
    "byjus.com": 3,
    "vedantu.com": 3,
    "coursera.org": 3,
    "udemy.com": 2,
    "docs.python.org": 6,
    "developer.mozilla.org": 5,
}

# Words/phrases that hint at the kind of resource the user wants.
_QUERY_INTENT_MARKERS: dict[str, str] = {
    "pdf": "PDF notes",
    "notes": "class notes",
    "tutorial": "tutorial",
    "video": "video lecture",
    "lecture": "video lecture",
    "mcq": "MCQ practice questions",
    "questions": "practice questions",
    "solutions": "solved examples",
    "formula": "formula sheet",
    "summary": "summary revision notes",
}

# Stopwords stripped when extracting the topic from the user's request.
_SEARCH_FILLERS = [
    "please search", "search the web for", "search for", "search",
    "find notes for", "find notes on", "find notes", "notes for",
    "notes on", "find",
    "need a", "need an", "need some", "need", "i want", "i need",
    "can you", "could you", "please find", "get me", "give me",
    "show me", "look up", "find me", "i am looking for", "i'm looking for",
    "looking for", "web search", "google", "study material for",
    "study material", "study resources for", "resources for",
    "help me with", "help with",
]


@dataclass
class SearchResult:
    """A single web-search result."""

    title: str
    url: str
    snippet: str = ""
    domain: str = ""
    rank_score: float = 0.0

    def as_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


class SearchEngine:
    """Handles query construction, web searching and browser actions."""

    def __init__(self, knowledge: KnowledgeBase) -> None:
        self.knowledge = knowledge
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": str(RUNTIME["search_user_agent"])})
        # A small pool of User-Agents is rotated so a single rate-limited UA
        # doesn't get the whole engine blocked.
        self._user_agents = [
            str(RUNTIME["search_user_agent"]),
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0",
        ]
        self._ua_index = 0
        self.last_results: list[SearchResult] = []
        # Small in-memory cache of extracted answers (query -> answer dict),
        # so asking the same question twice never re-fetches the web.
        self._answer_cache: dict[str, dict[str, str]] = {}

    def _rotate_user_agent(self) -> None:
        """Switch to the next User-Agent in the pool."""
        self._ua_index = (self._ua_index + 1) % len(self._user_agents)
        self._session.headers.update({"User-Agent": self._user_agents[self._ua_index]})

    # ------------------------------------------------------------------
    # Query construction
    # ------------------------------------------------------------------
    def build_query(self, user_text: str, resource_hint: Optional[str] = None) -> str:
        """Turn free-form user text into a clean search query.

        Examples
        --------
        >>> engine.build_query("Search Python loops")
        'Python loops'
        >>> engine.build_query("Find notes for Biology")
        'Biology class notes'
        """
        original = user_text.strip()
        original_lowered = original.lower()

        # 1) Detect the resource marker ("notes", "pdf", "tutorial", ...)
        #    from the ORIGINAL text so it is not lost with the filler words.
        marker = ""
        for word, suffix in _QUERY_INTENT_MARKERS.items():
            if re.search(rf"\b{re.escape(word)}\b", original_lowered):
                marker = suffix
                break
        if resource_hint:
            marker = _QUERY_INTENT_MARKERS.get(resource_hint.lower(), resource_hint)

        # 2) Strip filler words to extract the bare topic.
        cleaned = original
        lowered = original_lowered
        for filler in sorted(_SEARCH_FILLERS, key=len, reverse=True):
            if lowered.startswith(filler):
                cleaned = cleaned[len(filler):].strip()
                lowered = cleaned.lower()
                break

        # 3) Avoid redundancy: "find notes for biology" should become
        #    "Biology class notes", never "notes for biology class notes".
        if marker:
            marker_words = marker.lower().split()
            if any(word in lowered for word in marker_words):
                marker = ""

        query_parts = [part for part in (cleaned, marker) if part]
        query = " ".join(query_parts).strip()
        return query or "study resources"

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------
    def search(self, query: str, max_results: Optional[int] = None) -> list[SearchResult]:
        """Search the web for ``query`` and return ranked results.

        A **fallback chain** of live sources is tried in order:

          1. DuckDuckGo HTML
          2. Bing HTML
          3. Wikipedia search (as a last live source)

        Only when *every* live source fails does the engine fall back to the
        curated offline list. A single failing source is logged at DEBUG
        (not WARNING) — it is normal for one engine to be rate-limited while
        another works fine.
        """
        limit = int(max_results or RUNTIME.get("search_max_results", 8))
        engines = [
            ("duckduckgo", self._query_duckduckgo),
            ("bing", self._query_bing),
            ("wikipedia", self._query_wikipedia),
        ]
        results: list[SearchResult] = []
        failures: list[str] = []
        served_by = "offline"
        for attempt in range(2):  # one full retry with a rotated User-Agent
            for name, fetcher in engines:
                try:
                    live = fetcher(query, limit)
                    if live:
                        results = live
                        served_by = name
                        break
                    failures.append(f"{name}: empty")
                except (requests.RequestException, ValueError, ConnectionError) as exc:
                    failures.append(f"{name}: {exc}")
                    log.debug("Search source '%s' failed for %r: %s", name, query, exc)
            if results or attempt >= 1:
                break
            self._rotate_user_agent()

        if not results:
            log.warning("All live search sources failed for %r (%s); "
                        "using offline fallback.", query, "; ".join(failures))
            results = self._offline_fallback(query, limit)
        else:
            log.debug("Search for %r served by %s", query, served_by)

        self.last_results = results
        return results

    def _query_duckduckgo(self, query: str, limit: int) -> list[SearchResult]:
        """Hit DuckDuckGo's HTML endpoint and parse the results page."""
        url = "https://html.duckduckgo.com/html/"
        params = {"q": query, "kl": "us-en"}
        timeout = float(RUNTIME.get("search_timeout", 12))
        response = self._session.get(url, params=params, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        raw_results: list[SearchResult] = []

        for block in soup.select("div.result"):
            link = block.select_one("a.result__a")
            if link is None:
                continue
            href = link.get("href", "")
            # DuckDuckGo wraps real URLs in a redirect path.
            real_url = self._unwrap_url(href)
            if not real_url.startswith(("http://", "https://")):
                continue
            title = html_lib.unescape(link.get_text(" ", strip=True))
            snippet_node = block.select_one("a.result__snippet")
            snippet = html_lib.unescape(
                snippet_node.get_text(" ", strip=True) if snippet_node else ""
            )
            raw_results.append(
                SearchResult(title=title, url=real_url, snippet=snippet,
                             domain=self._domain_of(real_url))
            )

        ranked = self._rank(raw_results)
        return ranked[:limit]

    @staticmethod
    def _unwrap_url(duck_url: str) -> str:
        """Extract the real URL from DuckDuckGo's redirect link."""
        if "uddg=" in duck_url:
            from urllib.parse import unquote, urlparse, parse_qs
            parsed = parse_qs(urlparse(duck_url).query)
            if "uddg" in parsed and parsed["uddg"]:
                return unquote(parsed["uddg"][0])
        return duck_url

    def _query_bing(self, query: str, limit: int) -> list[SearchResult]:
        """Hit Bing's HTML endpoint and parse the organic results.

        Bing is the second live source in the fallback chain; its HTML is
        simpler than DuckDuckGo's and it rarely CAPTCHAs simple requests.
        """
        url = "https://www.bing.com/search"
        params = {"q": query}
        timeout = float(RUNTIME.get("search_timeout", 12))
        response = self._session.get(url, params=params, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        raw_results: list[SearchResult] = []
        seen_urls: set[str] = set()

        for block in soup.select("li.b_algo"):
            link = block.select_one("h2 a")
            if link is None:
                continue
            href = link.get("href", "")
            if not href.startswith(("http://", "https://")):
                continue
            if href in seen_urls:
                continue
            seen_urls.add(href)
            title = html_lib.unescape(link.get_text(" ", strip=True))
            snippet_node = block.select_one("p")
            snippet = html_lib.unescape(
                snippet_node.get_text(" ", strip=True) if snippet_node else ""
            )
            raw_results.append(SearchResult(
                title=title, url=href, snippet=snippet,
                domain=self._domain_of(href),
            ))

        return self._rank(raw_results)[:limit]

    def _query_wikipedia(self, query: str, limit: int) -> list[SearchResult]:
        """Use Wikipedia's search API as a last-resort live source.

        Returns article matches as search results — not a general web search,
        but still genuinely useful live results for study questions.
        """
        search_query = re.sub(
            r"^(what|who|when|where|which|why|how|is|are|was|were|does|do|did|"
            r"can|could|define|tell me about)\s+(?:the\s+|a\s+|an\s+)?",
            "", query.strip(), flags=re.IGNORECASE
        ).strip() or query
        timeout = float(RUNTIME.get("search_timeout", 12))
        response = self._session.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "list": "search",
                "srsearch": search_query, "format": "json", "srlimit": limit,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        hits = response.json().get("query", {}).get("search", [])
        from urllib.parse import quote
        results: list[SearchResult] = []
        for hit in hits[:limit]:
            title = hit.get("title", "")
            if not title:
                continue
            url = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
            results.append(SearchResult(
                title=f"Wikipedia — {title}",
                url=url,
                snippet=html_lib.unescape(
                    re.sub(r"<[^>]+>", "", hit.get("snippet", ""))
                ),
                domain="wikipedia.org",
            ))
        return self._rank(results)[:limit]

    # ------------------------------------------------------------------
    # Ranking & fallback
    # ------------------------------------------------------------------
    def _rank(self, results: list[SearchResult]) -> list[SearchResult]:
        """Score results: priority domains and keyword richness rank higher."""
        for result in results:
            score = 1.0
            domain = result.domain
            for trusted, bonus in PRIORITY_DOMAINS.items():
                if trusted in domain:
                    score += bonus
                    break
            if any(kw in result.title.lower() for kw in ("notes", "pdf", "tutorial",
                                                          "lecture", "documentation")):
                score += 1.0
            if result.snippet:
                score += 0.5
            result.rank_score = score
        results.sort(key=lambda r: r.rank_score, reverse=True)
        return results

    def _offline_fallback(self, query: str, limit: int) -> list[SearchResult]:
        """Return curated study links when the network is unavailable."""
        curated: list[SearchResult] = []
        subject = query.lower()
        curated.append(SearchResult(
            title="NCERT — National Council of Educational Research and Training",
            url="https://ncert.nic.in/textbook.php",
            snippet="Official NCERT textbooks and study material (India).",
            domain="ncert.nic.in", rank_score=10,
        ))
        curated.append(SearchResult(
            title="Khan Academy — free courses & practice",
            url="https://www.khanacademy.org/",
            snippet="Free world-class education for anyone, anywhere.",
            domain="khanacademy.org", rank_score=10,
        ))
        curated.append(SearchResult(
            title=f"Wikipedia — {query.title()}",
            url=f"https://en.wikipedia.org/wiki/{'_'.join(query.title().split())}",
            snippet="General overview and background reading.",
            domain="wikipedia.org", rank_score=9,
        ))
        if any(kw in subject for kw in ("python", "java", "html", "css", "javascript", "programming", "code")):
            curated.append(SearchResult(
                title="GeeksforGeeks — programming tutorials",
                url="https://www.geeksforgeeks.org/",
                snippet="Computer science and programming articles, quizzes and practice.",
                domain="geeksforgeeks.org", rank_score=9,
            ))
            curated.append(SearchResult(
                title="W3Schools — web development tutorials",
                url="https://www.w3schools.com/",
                snippet="Free tutorials and references for HTML, CSS, JS, Python and more.",
                domain="w3schools.com", rank_score=9,
            ))
            curated.append(SearchResult(
                title="Python.org — official documentation",
                url="https://docs.python.org/3/tutorial/",
                snippet="The Python language reference and tutorial.",
                domain="docs.python.org", rank_score=9,
            ))
        curated.append(SearchResult(
            title="YouTube — educational video search",
            url="https://www.youtube.com/results?search_query=" + query.replace(" ", "+"),
            snippet="Video lectures and explainers on any topic.",
            domain="youtube.com", rank_score=8,
        ))
        curated.append(SearchResult(
            title="Google Scholar — academic papers",
            url="https://scholar.google.com/scholar?q=" + query.replace(" ", "+"),
            snippet="Scholarly articles and research papers.",
            domain="scholar.google.com", rank_score=8,
        ))
        return self._rank(curated)[:limit]

    # ------------------------------------------------------------------
    # Web-answer extraction: pull the passage that answers a query
    # ------------------------------------------------------------------
    def extract_answer(self, query: str,
                       results: Optional[list[SearchResult]] = None,
                       max_fetch: int = 2) -> Optional[dict[str, str]]:
        """Answer ``query`` by extracting from the web.

        Strategy (in order):
          1. Fetch the top search-result pages and pull the best passage.
          2. If that fails — or the search engine was blocked/offline — ask
             Wikipedia's summary API, which returns clean, reliable prose
             for factual questions.

        Returns a dict ``{"passage", "title", "url"}`` on success, or
        ``None`` when nothing yields a confident answer. The caller can then
        fall back to showing the plain results list.

        Parameters
        ----------
        query : str
            The search query / question being answered.
        results : list[SearchResult] | None
            Pre-fetched results (defaults to a fresh search).
        max_fetch : int
            How many top results to fetch and scan (default 2 — keeps the
            interaction fast and polite to the sites).
        """
        results = results if results is not None else self.search(query)
        keywords = self._query_keywords(query)
        if not keywords:
            return None

        cache_key = " ".join(keywords)
        cached = self._answer_cache.get(cache_key)
        if cached:
            return dict(cached)

        for result in results[:max_fetch]:
            try:
                text = self.fetch_page_text(result.url)
            except (requests.RequestException, ValueError, OSError) as exc:
                log.debug("Skipping %s: %s", result.url, exc)
                continue
            if not text or self._is_boilerplate(text):
                log.debug("Skipping %s (empty or boilerplate)", result.url)
                continue
            passage = self._best_passage(text, keywords)
            if passage and len(passage) >= 40:
                log.info("Extracted answer from %s", result.url)
                answer = {
                    "passage": passage,
                    "title": result.title,
                    "url": result.url,
                }
                self._remember_answer(cache_key, answer)
                return answer

        # Second source: Wikipedia's summary API (reliable, no CAPTCHA).
        wiki = self.wikipedia_summary(query)
        if wiki:
            log.info("Answered via Wikipedia summary for %r", query)
            self._remember_answer(cache_key, wiki)
            return wiki
        return None

    def extract_brief(self, query: str,
                      results: Optional[list[SearchResult]] = None,
                      max_fetch: int = 2,
                      max_passages: int = 2) -> Optional[dict[str, object]]:
        """Answer ``query`` with a short, readable brief (not just a link).

        Fetches the top pages, extracts the paragraphs richest in the
        query's keywords, and returns up to ``max_passages`` of them so the
        user gets a real inline explanation — plus the source title/URL.

        Returns a dict::

            {"title": str, "url": str, "passages": [str, ...]}

        or ``None`` when nothing could be extracted.
        """
        results = results if results is not None else self.search(query)
        keywords = self._query_keywords(query)
        if not keywords:
            return None

        # 1. Prefer Wikipedia's clean summary (reliable, prose-only intro).
        wiki = self.wikipedia_summary(query)
        if wiki:
            passage = wiki.get("passage", "")
            if passage and len(passage) >= 40:
                return {
                    "title": wiki.get("title", ""),
                    "url": wiki.get("url", ""),
                    "passages": [passage],
                }

        # 2. Otherwise scrape the top pages for keyword-rich passages.
        passages: list[str] = []
        chosen_title = ""
        chosen_url = ""
        for result in results[:max_fetch]:
            try:
                text = self.fetch_page_text(result.url)
            except (requests.RequestException, ValueError, OSError) as exc:
                log.debug("Skipping %s: %s", result.url, exc)
                continue
            if not text or self._is_boilerplate(text):
                continue
            found = self._best_passages(text, keywords, max_passages - len(passages))
            for passage in found:
                if passage not in passages:
                    passages.append(passage)
            if found and not chosen_title:
                chosen_title = result.title
                chosen_url = result.url
            if len(passages) >= max_passages:
                break

        if not passages:
            return None
        return {"title": chosen_title, "url": chosen_url, "passages": passages}

    def _best_passages(self, text: str, keywords: list[str],
                       count: int = 2, min_chars: int = 60) -> list[str]:
        """Return the top ``count`` keyword-rich prose passages.

        Like :meth:`_best_passage` but collects several distinct passages so
        the answer reads as a brief instead of a single clipped sentence.
        Each passage is cleaned: boilerplate prefixes are stripped and the
        text is trimmed to the sentences richest in the query's keywords.
        """
        paragraphs = [p.strip() for p in re.split(r"\n+", text)
                      if len(p.strip()) > min_chars and self._looks_like_prose(p.strip())]
        scored = []
        for paragraph in paragraphs:
            # Skip coordinate/infobox junk templates.
            if re.search(r"coordinates?\s*[:·]\s*\d|geo\s*[:·]", paragraph, re.I):
                continue
            score = self._score_text(paragraph, keywords)
            if score >= 1.0:
                scored.append((score, paragraph))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        picked: list[str] = []
        seen: set[str] = set()
        for _score, paragraph in scored:
            cleaned = self._clean_passage(paragraph, keywords)
            if not cleaned:
                continue
            key = cleaned[:120]
            if key in seen:
                continue
            seen.add(key)
            picked.append(cleaned)
            if len(picked) >= count:
                break
        return picked

    def _clean_passage(self, paragraph: str, keywords: list[str]) -> str:
        """Trim a passage to the sentences richest in the query keywords.

        Strips Wikipedia-style boilerplate ("From Wikipedia, the free
        encyclopedia", "For other uses, see ...") and returns the best
        1-3 consecutive sentences that actually mention the keywords.
        """
        text = paragraph.strip()
        # Drop leading boilerplate lines.
        lowered = text.lower()
        for marker in ("from wikipedia, the free encyclopedia",
                       "this article is about", "for other uses, see",
                       "not to be confused with"):
            idx = lowered.find(marker)
            if idx >= 0 and idx < 120:
                end = text.find(".", idx)
                if end != -1:
                    text = text[end + 1:].strip()
                    lowered = text.lower()

        # Strip footnote/reference markers like "[18]" or "[citation needed]".
        text = re.sub(r"\[\s*\d+\s*\]", "", text)
        text = re.sub(r"\[ citation needed \]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentences:
            return text[:420]

        # Penalise caption/nav-like sentences, boost definitional ones
        # ("X is a...", "X is the...", "X are...", "refers to...").
        caption_penalty = ("image", "figure", "schematic", "timeline",
                           "see also", "lifespan", "incumbent", "photo of")
        definition_boost = (" is a ", " is the ", " are the ", " refers to ",
                            " is an ", " was a ", " was the ", " is one of ")

        scored = []
        for i, s in enumerate(sentences):
            low = s.lower()
            base = self._score_text(s, keywords)
            if any(word in low for word in caption_penalty):
                base -= 2.0
            if any(word in low for word in definition_boost):
                base += 1.5
            scored.append((base, i, s))
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_idx, _best_s = scored[0]
        if best_score < 1.0:
            # No keyword-rich sentence — return the start of the passage.
            return text[:420]

        # Return the best sentence plus the one after it (a mini brief).
        window = sentences[best_idx:best_idx + 2]
        brief = " ".join(window)
        if len(brief) > 420:
            brief = brief[:417] + "…"
        return brief

    def _remember_answer(self, key: str, answer: dict[str, str]) -> None:
        """Cache an extracted answer, keeping the cache bounded."""
        if len(self._answer_cache) >= 50:  # oldest entry evicted
            try:
                self._answer_cache.pop(next(iter(self._answer_cache)))
            except StopIteration:
                pass
        self._answer_cache[key] = answer

    def wikipedia_summary(self, query: str) -> Optional[dict[str, str]]:
        """Fetch a concise answer from Wikipedia's summary API.

        Uses the search API to find the best-matching article, then the REST
        summary endpoint for a clean one-paragraph intro. The passage is
        trimmed to the part richest in the query's keywords.

        Returns ``None`` on any failure (network, missing article, ...).
        """
        import time
        timeout = float(RUNTIME.get("search_timeout", 12))
        for attempt in range(2):  # one retry with a short backoff
            result = self._wikipedia_summary_once(query, timeout)
            if result is not None:
                return result
            time.sleep(0.5 * (attempt + 1))
        return None

    def _wikipedia_summary_once(self, query: str, timeout: float) -> Optional[dict[str, str]]:
        """Single attempt at fetching a Wikipedia summary (see wikipedia_summary)."""
        # Strip question prefixes so the article search matches the TOPIC,
        # not the phrasing: "who wrote harry potter" -> "harry potter".
        search_query = re.sub(
            r"^(what|who|when|where|which|why|how|is|are|was|were|does|do|did|"
            r"can|could|define|tell me about|whats)\s+(?:the\s+|a\s+|an\s+)?",
            "", query.strip(), flags=re.IGNORECASE
        ).strip()
        if not search_query:
            search_query = query
        try:
            search_response = self._session.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query", "list": "search",
                    "srsearch": search_query, "format": "json", "srlimit": 3,
                },
                timeout=timeout,
            )
            search_response.raise_for_status()
            hits = search_response.json().get("query", {}).get("search", [])
            if not hits:
                return None

            # Prefer the hit whose title best matches the topic words, so a
            # question about "harry potter" picks the novel, not the film
            # series spin-off. Fall back to the first hit.
            title = hits[0]["title"]
            topic_words = [w for w in re.findall(r"[a-zA-Z]{3,}", search_query.lower())
                           if w not in {"the", "and", "what", "who", "whats"}]
            if topic_words:
                best_title, best_hits = title, -1
                for hit in hits[:3]:
                    hit_lower = hit["title"].lower()
                    count = sum(1 for w in topic_words if w in hit_lower)
                    # Penalise parenthetical disambiguators like "(film series)".
                    if "(" in hit_lower:
                        count -= 0.5
                    if count > best_hits:
                        best_hits = count
                        best_title = hit["title"]
                title = best_title

            from urllib.parse import quote
            summary_response = self._session.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}",
                timeout=timeout,
            )
            summary_response.raise_for_status()
            data = summary_response.json()
            extract = data.get("extract", "")
            if not extract or self._is_boilerplate(extract):
                return None

            keywords = self._query_keywords(query)
            passage = self._best_passage(extract, keywords) if keywords else None
            if not passage or len(passage) < 40:
                passage = extract

            page_url = (
                data.get("content_urls", {}).get("desktop", {}).get("page")
                or f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            )
            # displaytitle can contain HTML (<i>, <span>...) — keep it plain.
            display = re.sub(r"<[^>]+>", "", data.get("displaytitle", title))
            return {
                "passage": passage,
                "title": f"Wikipedia — {display}",
                "url": page_url,
            }
        except (requests.RequestException, ValueError, KeyError) as exc:
            log.debug("Wikipedia summary failed: %s", exc)
            return None

    def fetch_page_text(self, url: str, max_chars: int = 12000) -> str:
        """Download ``url`` and return its cleaned main text.

        Scripts, styles, navigation, headers, footers and ads are stripped;
        whitespace is collapsed. ``max_chars`` caps the returned text so
        enormous pages stay fast to scan.
        """
        timeout = float(RUNTIME.get("search_timeout", 12))
        response = self._session.get(url, timeout=timeout, stream=True)
        # Memory guard: refuse enormous pages before parsing them.
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > 2 * 1024 * 1024:
            response.close()
            raise ValueError(f"page too large ({content_length} bytes)")
        response.raise_for_status()
        html = response.text[:2 * 1024 * 1024]  # hard cap before parsing

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "nav", "header",
                         "footer", "aside", "form", "iframe", "table"]):
            # Tables are almost always data/infobox noise (e.g. Wikipedia's
            # infobox) rather than the answer passage itself.
            tag.decompose()

        # Prefer the main content container when present; else whole body.
        main = (soup.find("main") or soup.find("article")
                or soup.find("div", {"class": re.compile(r"content|article|body", re.I)})
                or soup.body)
        if main is None:
            main = soup

        text = main.get_text(" ", strip=True)
        text = re.sub(r"\s+", " ", text)
        return text[:max_chars]

    @staticmethod
    def _query_keywords(query: str) -> list[str]:
        """Extract meaningful keywords from a query for passage scoring.

        Drops stopwords and very short tokens, keeps the longest 6.
        """
        stopwords = {
            "the", "and", "for", "with", "what", "why", "how", "who", "when",
            "where", "which", "that", "this", "these", "those", "does", "do",
            "did", "is", "are", "was", "were", "can", "could", "would", "should",
            "about", "from", "into", "whats", "difference", "between", "define",
            "meaning", "of", "in", "on", "at", "to", "by", "as", "an", "a",
            "it", "its", "or", "but", "not", "no", "up", "out", "off", "than",
            "then", "also", "very", "more", "most", "has", "have", "had", "be",
            "been", "being", "use", "used", "using", "get", "got", "give",
            "whats", "tell", "me", "my", "your", "you", "we", "they", "he", "she",
        }
        words = re.findall(r"[a-zA-Z0-9]{2,}", query.lower())
        keywords = [w for w in words if w not in stopwords and len(w) >= 4]
        return sorted(set(keywords), key=len, reverse=True)[:6]

    def _best_passage(self, text: str, keywords: list[str],
                      max_passage: int = 420) -> Optional[str]:
        """Find the paragraph/sentence in ``text`` richest in keywords.

        Paragraphs are scored by keyword frequency (with a small bonus for
        hitting several distinct keywords); if the winning paragraph is too
        long it is trimmed to its best sentence. Returns ``None`` when no
        passage reaches the confidence threshold.
        """
        paragraphs = [p.strip() for p in re.split(r"\n+", text)
                      if len(p.strip()) > 40 and self._looks_like_prose(p.strip())]
        best_paragraph = ""
        best_score = 0.0
        for paragraph in paragraphs:
            score = self._score_text(paragraph, keywords)
            if score > best_score:
                best_score = score
                best_paragraph = paragraph

        if not best_paragraph or best_score < 1.0:
            return None

        # Skip junk templates like "Coordinates: 15°47′38″S …" which appear
        # at the top of many Wikipedia-style articles.
        if re.search(r"coordinates?\s*[:·]\s*\d|geo\s*[:·]", best_paragraph, re.I):
            return None

        if len(best_paragraph) <= max_passage:
            return best_paragraph

        # Trim a long paragraph to its best sentence.
        sentences = re.split(r"(?<=[.!?])\s+", best_paragraph)
        best_sentence, best_sentence_score = "", 0.0
        for sentence in sentences:
            score = self._score_text(sentence, keywords)
            if score > best_sentence_score:
                best_sentence_score = score
                best_sentence = sentence
        if best_sentence and len(best_sentence) >= 40:
            return best_sentence
        return best_paragraph[:max_passage] + "…"

    @staticmethod
    def _score_text(text: str, keywords: list[str]) -> float:
        """Score how well a passage covers the query keywords."""
        lowered = text.lower()
        distinct = sum(1 for kw in keywords if kw in lowered)
        # Small bonus for repeated occurrences of the top keyword.
        top = keywords[0] if keywords else ""
        frequency = lowered.count(top) if top else 0
        return distinct + (0.1 * min(frequency, 5))

    @staticmethod
    def _is_boilerplate(text: str) -> bool:
        """Detect error/loading/blocker pages that contain no real answer.

        These pages (CAPTCHA walls, 'enable JavaScript' notices, network
        error screens) look like prose but carry no useful content, so the
        extractor must skip them instead of quoting an error message.
        """
        lowered = text.lower()
        markers = [
            "couldn't load", "could not load", "required part of this site",
            "browser extension", "ad blockers", "check your connection",
            "enable javascript", "javascript is required", "access denied",
            "captcha", "complete the following challenge", "verify you are human",
            "404", "page not found", "this page doesn't exist", "unfortunately",
            "turn on javascript", "please wait while we", "confirm this search",
        ]
        hits = sum(1 for marker in markers if marker in lowered)
        # One decisive marker on a short page, or two markers anywhere →
        # treat as boilerplate.
        return (hits >= 1 and len(text) < 4000) or hits >= 2

    @staticmethod
    def _looks_like_prose(text: str) -> bool:
        """Heuristic filter for readable sentences vs. nav/infobox junk.

        A passage counts as prose when it contains several words, ends like
        a sentence (or has a verb-ish density) and isn't mostly labels or
        punctuation. Cheap but effective at skipping menu/infobox noise.
        """
        words = re.findall(r"[a-zA-Z]{2,}", text)
        if len(words) < 5:
            return False
        # A sentence usually ends with . ! ? or a closing bracket.
        if re.search(r"[.!?]\s*$", text.strip()):
            return True
        # Fall back: at least 60% of tokens must be alphabetic words.
        tokens = text.split()
        if not tokens:
            return False
        alpha = sum(1 for token in tokens if re.fullmatch(r"[a-zA-Z]+", token))
        return alpha / len(tokens) >= 0.6

    # ------------------------------------------------------------------
    # Browser integration
    # ------------------------------------------------------------------
    @staticmethod
    def open_in_browser(url: str) -> bool:
        """Open ``url`` in the user's default web browser.

        Returns ``True`` if the browser was (or appeared to be) launched.
        """
        try:
            webbrowser.open(url, new=2)
            log.info("Opened browser: %s", url)
            return True
        except webbrowser.Error as exc:
            log.error("Could not open browser: %s", exc)
            return False

    def open_result(self, index: int) -> bool:
        """Open the result at ``index`` (1-based) from the last search."""
        if not self.last_results:
            return False
        if index < 1 or index > len(self.last_results):
            return False
        return self.open_in_browser(self.last_results[index - 1].url)

    # ------------------------------------------------------------------
    # Formatting helpers (used by chatbot)
    # ------------------------------------------------------------------
    @staticmethod
    def _domain_of(url: str) -> str:
        from urllib.parse import urlparse
        return urlparse(url).netloc.lower().replace("www.", "")

    def format_results(self, results: list[SearchResult], query: str) -> str:
        """Render results as a numbered list for the console."""
        lines = [
            f"🔎 Top results for **{query}** — {len(results)} found:",
            "",
        ]
        for i, result in enumerate(results, start=1):
            lines.append(f"{i}. **{result.title}**")
            lines.append(f"   {result.url}")
            if result.snippet:
                snippet = result.snippet if len(result.snippet) <= 140 else result.snippet[:137] + "…"
                lines.append(f"   {snippet}")
            lines.append("")
        lines.append(
            "💡 Tip: type `open <number>` to open a result in your browser, "
            "or `open top` for the best one."
        )
        return "\n".join(lines)
