// Global variables
let wsConnection = null;
let timerInterval = null;
let activeReportContent = ""; // Stores raw markdown report
let globalSourcesList = [];   // Stores unique sources found during search

// DOM Elements
const sidebar = document.getElementById("sidebar");
const appContainer = document.getElementById("app-container");
const mainContent = document.getElementById("main-content");
const historyList = document.getElementById("history-list");

const btnSidebarCollapse = document.getElementById("btn-sidebar-collapse");
const btnSidebarExpand = document.getElementById("btn-sidebar-expand");

const viewSearch = document.getElementById("view-search");
const viewRunning = document.getElementById("view-running");
const viewResult = document.getElementById("view-result");

const searchForm = document.getElementById("search-form");
const searchInput = document.getElementById("search-input");
const btnSubmit = document.getElementById("btn-submit");

const runningQueryDisplay = document.getElementById("running-query-display");
const runningTimer = document.getElementById("running-timer");
const consoleLogs = document.getElementById("console-logs");

const researchersBranch = document.getElementById("researchers-branch");
const activeSourcesBar = document.getElementById("active-sources-bar");
const activeSourcesStack = document.getElementById("active-sources-stack");

const reportSourcesSection = document.getElementById("report-sources-section");
const reportSourcesAvatars = document.getElementById("report-sources-avatars");
const reportSourcesGrid = document.getElementById("report-sources-grid");
const reportSourcesTrigger = document.getElementById("report-sources-trigger");
const sourcesCountText = document.getElementById("sources-count");

const reportOutputRendered = document.getElementById("report-output-rendered");
const btnNewSearch = document.getElementById("btn-new-search");
const btnCopyMarkdown = document.getElementById("btn-copy-markdown");
const btnPrint = document.getElementById("btn-print");

const toast = document.getElementById("toast");
const toastIcon = document.getElementById("toast-icon");
const toastMessage = document.getElementById("toast-message");

// Graph node references (Planner, Arbitrator, Synthesizer)
const nodes = {
    planner: document.getElementById("node-planner"),
    arbitrator: document.getElementById("node-arbitrator"),
    synthesizer: document.getElementById("node-synthesizer")
};

// ==========================================================================
//  INITIALIZATION & SIDEBAR CONTROL
// ==========================================================================
document.addEventListener("DOMContentLoaded", () => {
    loadHistory();
    setupEventListeners();
});

function setupEventListeners() {
    // Sidebar Collapsing
    btnSidebarCollapse.addEventListener("click", () => {
        sidebar.classList.add("collapsed");
        btnSidebarExpand.classList.remove("hidden");
    });

    btnSidebarExpand.addEventListener("click", () => {
        sidebar.classList.remove("collapsed");
        btnSidebarExpand.classList.add("hidden");
    });

    // Form submission starts the WebSocket research flow
    searchForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const query = searchInput.value.trim();
        if (query) {
            startResearch(query);
        }
    });

    // Clicking sample suggestions populates search and submits
    document.querySelectorAll(".suggestion-chip").forEach(chip => {
        chip.addEventListener("click", () => {
            const query = chip.getAttribute("data-query");
            searchInput.value = query;
            startResearch(query);
        });
    });

    // Back / New Research button
    btnNewSearch.addEventListener("click", () => {
        switchView("search");
        searchInput.value = "";
    });

    // Copy Raw Markdown
    btnCopyMarkdown.addEventListener("click", () => {
        if (!activeReportContent) return;
        navigator.clipboard.writeText(activeReportContent)
            .then(() => showToast("success", "Markdown copied to clipboard!"))
            .catch(err => showToast("error", "Failed to copy: " + err));
    });

    // Print / PDF Export
    btnPrint.addEventListener("click", () => {
        window.print();
    });

    // Toggle sources grid collapse
    reportSourcesTrigger.addEventListener("click", () => {
        reportSourcesTrigger.classList.toggle("open");
        reportSourcesGrid.classList.toggle("hidden");
    });
}

// Load previous reports from the FastAPI directory
async function loadHistory() {
    try {
        const response = await fetch("/api/history");
        if (!response.ok) throw new Error("Failed to fetch history list");
        
        const files = await response.json();
        
        if (files.length === 0) {
            historyList.innerHTML = `<div class="history-empty">No dossiers found. Start research to generate one!</div>`;
            return;
        }

        historyList.innerHTML = "";
        files.forEach(file => {
            const date = new Date(file.createdTime * 1000).toLocaleDateString(undefined, {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
            const sizeKB = (file.sizeBytes / 1024).toFixed(1);
            
            const item = document.createElement("div");
            item.className = "history-item";
            item.innerHTML = `
                <div class="history-title" title="${file.title}">${file.title}</div>
                <div class="history-meta">
                    <span>${date}</span>
                    <span>${sizeKB} KB</span>
                </div>
            `;
            item.addEventListener("click", () => loadSavedReport(file.filename));
            historyList.appendChild(item);
        });

    } catch (error) {
        console.error("Error loading history:", error);
        historyList.innerHTML = `<div class="history-empty" style="color:var(--danger)">Error loading history.</div>`;
    }
}

// Fetch report content from API and display it
async function loadSavedReport(filename) {
    try {
        const response = await fetch(`/api/history/load?filename=${encodeURIComponent(filename)}`);
        if (!response.ok) throw new Error("Failed to load report file");
        
        const data = await response.json();
        activeReportContent = data.content;
        
        renderReport(activeReportContent);
        switchView("result");
        showToast("success", "Dossier loaded!");
    } catch (error) {
        showToast("error", "Error loading report: " + error.message);
    }
}

// ==========================================================================
//  WEBSOCKET CONNECTION & RESEARCH FLOW
// ==========================================================================
function startResearch(query) {
    // Reset timeline and UI
    resetTimeline();
    consoleLogs.innerHTML = "";
    runningQueryDisplay.textContent = `Topic: "${query}"`;
    globalSourcesList = [];
    activeSourcesStack.innerHTML = "";
    activeSourcesBar.classList.add("hidden");
    
    // Reset researcher container to placeholder
    researchersBranch.innerHTML = `<div class="researcher-placeholder">Planner is decomposing query...</div>`;
    
    // Switch to active running panel
    switchView("running");
    appendLog("system", `Connecting to ResearchMind core pipeline...`);
    
    // Start timer clock
    startTimer();

    // Establish WebSocket connection
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws/research`;
    
    wsConnection = new WebSocket(wsUrl);

    wsConnection.onopen = () => {
        appendLog("system", `Connection opened. Triggering research agents...`);
        wsConnection.send(JSON.stringify({ query: query }));
    };

    wsConnection.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleProgressMessage(data);
    };

    wsConnection.onerror = (error) => {
        console.error("WebSocket Error:", error);
        appendLog("error", `API Connection Error. Verify server log.`);
        showToast("error", "WebSocket connection failure.");
        stopTimer();
    };

    wsConnection.onclose = () => {
        appendLog("system", `Connection closed.`);
        stopTimer();
    };
}

// Process incoming progress messages and update timeline & logs
function handleProgressMessage(data) {
    const event = data.event;
    const message = data.message;
    const details = data.data || {};

    switch (event) {
        // Node 1: Planner
        case "planner_start":
            setTimelineState("planner", "active");
            appendLog("planner", `[Planner] ${message}`);
            break;
            
        case "planner_end":
            setTimelineState("planner", "completed");
            appendLog("planner", `[Planner] ${message}`);
            
            // Build the parallel researcher cards
            if (details.subqueries && details.subqueries.length > 0) {
                researchersBranch.innerHTML = "";
                details.subqueries.forEach((sq, i) => {
                    const card = document.createElement("div");
                    card.className = "subquery-card idle";
                    card.id = `subquery-card-${i}`;
                    card.innerHTML = `
                        <div class="sq-card-header">
                            <span class="sq-badge">Subquery 0${i + 1}</span>
                            <span class="sq-status-icon" id="sq-status-${i}"><i class="fa-solid fa-circle-dot"></i></span>
                        </div>
                        <div class="sq-card-text" title="${sq}">${sq}</div>
                        <div class="sq-card-sources" id="sq-sources-${i}"></div>
                    `;
                    researchersBranch.appendChild(card);
                    appendLog("planner", `  └─ Subquery ${i + 1}: "${sq}"`);
                });
            }
            break;

        // Pacing delay
        case "pacing_start":
            appendLog("pacing", `[Pacing] ${message}`);
            break;

        // Node 2: Researchers
        case "researcher_start":
            // Highlight the running subquery card
            updateSubqueryCardState(details.index, "active");
            appendLog("researcher", `\n=== SUBQUERY ${details.index + 1} (Attempt ${details.retry + 1}) ===`);
            appendLog("researcher", `[Researchers] ${message}`);
            break;

        case "researcher_progress":
            appendLog("researcher", `  └─ ${message}`);
            // Check if there are sources sent in real-time
            if (details.sources && details.sources.length > 0) {
                // Find current active subquery card index based on active card in UI
                const activeIdx = getActiveSubqueryIndex();
                if (activeIdx !== -1) {
                    addSourcesToSubquery(activeIdx, details.sources);
                }
                updateGlobalSources(details.sources);
            }
            break;

        case "researcher_end":
            // Mark subquery as completed
            const completedIdx = getActiveSubqueryIndex();
            if (completedIdx !== -1) {
                updateSubqueryCardState(completedIdx, "completed");
                if (details.sources) {
                    addSourcesToSubquery(completedIdx, details.sources);
                    updateGlobalSources(details.sources);
                }
            }
            appendLog("researcher", `[Researchers] ${message}`);
            break;

        // Node 3: Arbitrator
        case "arbitrator_start":
            setTimelineState("arbitrator", "active");
            appendLog("arbitrator", `[Arbitrator] ${message}`);
            break;

        case "arbitrator_end":
            const accepted = details.accepted;
            if (accepted) {
                setTimelineState("arbitrator", "completed");
                appendLog("arbitrator", `[Arbitrator] Verdict: ACCEPTED — ${message}`);
            } else {
                setTimelineState("arbitrator", "idle");
                // Find active card and mark as rejected
                const activeIdx = getActiveSubqueryIndex();
                if (activeIdx !== -1) {
                    updateSubqueryCardState(activeIdx, "rejected");
                }
                appendLog("arbitrator", `[Arbitrator] Verdict: REJECTED — ${message}`);
            }
            break;

        // Node 4: Synthesizer
        case "synthesizer_start":
            setTimelineState("synthesizer", "active");
            appendLog("synthesizer", `[Synthesizer] ${message}`);
            break;

        case "synthesizer_end":
            setTimelineState("synthesizer", "completed");
            appendLog("synthesizer", `[Synthesizer] ${message}`);
            break;

        // Error Handlers
        case "error":
            appendLog("error", `CRITICAL ERROR: ${message}`);
            stopTimer();
            showToast("error", `Pipeline failed: ${message}`);
            break;

        // Completion
        case "complete":
            stopTimer();
            activeReportContent = data.report;
            
            renderReport(activeReportContent);
            
            // Switch views and refresh history list
            setTimeout(() => {
                switchView("result");
                showToast("success", "Synthesis complete!");
                loadHistory();
            }, 1000);
            break;
            
        default:
            appendLog("system", message);
    }
}

// ==========================================================================
//  SOURCES ENGINE (Real-time and Accordion Grid)
// ==========================================================================
function updateGlobalSources(urls) {
    if (!urls || urls.length === 0) return;
    
    activeSourcesBar.classList.remove("hidden");
    
    urls.forEach(url => {
        if (!url || typeof url !== "string") return;
        try {
            const domain = new URL(url).hostname;
            if (globalSourcesList.includes(url)) return;
            
            globalSourcesList.push(url);
            
            // Add overlapping favicon to active sources stack bar
            const badge = document.createElement("div");
            badge.className = "source-badge-circle";
            badge.title = domain;
            badge.setAttribute("onclick", `window.open('${url}', '_blank')`);
            badge.innerHTML = `<img src="https://www.google.com/s2/favicons?domain=${domain}&sz=32" alt="${domain}" onerror="this.src='https://cdn-icons-png.flaticon.com/512/1249/1249379.png';">`;
            activeSourcesStack.appendChild(badge);
        } catch(e) {}
    });
}

function addSourcesToSubquery(cardIndex, urls) {
    const sourcesContainer = document.getElementById(`sq-sources-${cardIndex}`);
    if (!sourcesContainer || !urls || urls.length === 0) return;
    
    sourcesContainer.innerHTML = "";
    
    const container = document.createElement("div");
    container.className = "stacked-avatars";
    
    const uniqueDomains = [];
    urls.forEach(url => {
        try {
            const domain = new URL(url).hostname;
            if (uniqueDomains.includes(domain)) return;
            uniqueDomains.push(domain);
            
            const img = document.createElement("img");
            img.src = `https://www.google.com/s2/favicons?domain=${domain}&sz=32`;
            img.alt = domain;
            img.title = domain;
            img.onerror = function() { this.src = 'https://cdn-icons-png.flaticon.com/512/1249/1249379.png'; };
            container.appendChild(img);
        } catch(e) {}
    });
    
    sourcesContainer.appendChild(container);
}

function getActiveSubqueryIndex() {
    const cards = researchersBranch.querySelectorAll(".subquery-card");
    for (let i = 0; i < cards.length; i++) {
        if (cards[i].classList.contains("active")) {
            return i;
        }
    }
    // Fallback: get first idle card
    for (let i = 0; i < cards.length; i++) {
        if (cards[i].classList.contains("idle")) {
            return i;
        }
    }
    return -1;
}

function updateSubqueryCardState(index, state) {
    const card = document.getElementById(`subquery-card-${index}`);
    if (!card) return;
    
    card.className = `subquery-card ${state}`;
    const statusIcon = document.getElementById(`sq-status-${index}`);
    if (statusIcon) {
        if (state === "active") {
            statusIcon.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i>`;
        } else if (state === "completed") {
            statusIcon.innerHTML = `<i class="fa-solid fa-circle-check"></i>`;
        } else if (state === "rejected") {
            statusIcon.innerHTML = `<i class="fa-solid fa-circle-xmark"></i>`;
        } else {
            statusIcon.innerHTML = `<i class="fa-solid fa-circle-dot"></i>`;
        }
    }
}

// ==========================================================================
//  REPORT RENDER & CITATION MARK PARSER
// ==========================================================================
function renderReport(markdown) {
    // 1. Render main markdown body using Marked
    let renderedHtml = marked.parse(markdown);
    
    // 2. Turn standard citations [1] or [1, 2] into beautiful interactive citation badges
    renderedHtml = renderedHtml.replace(/\[(\d+)\]/g, '<span class="citation-badge" title="Jump to Source $1" onclick="scrollToSource($1)">$1</span>');
    renderedHtml = renderedHtml.replace(/\[(\d+),\s*(\d+)\]/g, '<span class="citation-badge" title="Source $1" onclick="scrollToSource($1)">$1</span><span class="citation-badge" title="Source $2" onclick="scrollToSource($2)">$2</span>');
    
    reportOutputRendered.innerHTML = renderedHtml;
    
    // 3. Extract bibliography links and generate ChatGPT style sources header
    extractAndBuildReportSources(markdown);
}

function scrollToSource(number) {
    const reportPanel = viewResult;
    const bodyText = reportOutputRendered;
    // Find matching link index or lists of links at the end
    const sourcesHeader = bodyText.querySelector("h2:last-of-type, h2:nth-last-of-type(2)");
    if (sourcesHeader) {
        sourcesHeader.scrollIntoView({ behavior: 'smooth' });
    }
}

function extractAndBuildReportSources(markdown) {
    // Extract unique URL links from the markdown bibliography
    const urlRegex = /(https?:\/\/[^\s\)]+)/g;
    const matches = markdown.match(urlRegex) || [];
    const uniqueUrls = [...new Set(matches)];
    
    if (uniqueUrls.length === 0) {
        reportSourcesSection.classList.add("hidden");
        return;
    }
    
    reportSourcesSection.classList.remove("hidden");
    sourcesCountText.textContent = uniqueUrls.length;
    reportSourcesAvatars.innerHTML = "";
    reportSourcesGrid.innerHTML = "";
    
    const uniqueDomains = [];
    
    uniqueUrls.forEach(url => {
        try {
            const domain = new URL(url).hostname;
            
            // Build overlapping avatars
            if (!uniqueDomains.includes(domain)) {
                uniqueDomains.push(domain);
                const avatar = document.createElement("img");
                avatar.src = `https://www.google.com/s2/favicons?domain=${domain}&sz=32`;
                avatar.alt = domain;
                avatar.title = domain;
                avatar.onerror = function() { this.src = 'https://cdn-icons-png.flaticon.com/512/1249/1249379.png'; };
                reportSourcesAvatars.appendChild(avatar);
            }
            
            // Build dropdown card
            const card = document.createElement("a");
            card.href = url;
            card.target = "_blank";
            card.className = "source-card";
            card.innerHTML = `
                <img src="https://www.google.com/s2/favicons?domain=${domain}&sz=32" alt="${domain}" onerror="this.src='https://cdn-icons-png.flaticon.com/512/1249/1249379.png';">
                <div class="source-card-info">
                    <span class="source-card-domain">${domain}</span>
                    <span class="source-card-url" title="${url}">${url}</span>
                </div>
            `;
            reportSourcesGrid.appendChild(card);
        } catch(e) {}
    });
}

// ==========================================================================
//  UI UTILITY FUNCTIONS
// ==========================================================================
function switchView(viewName) {
    viewSearch.classList.add("hidden");
    viewRunning.classList.add("hidden");
    viewResult.classList.add("hidden");

    if (viewName === "search") {
        viewSearch.classList.remove("hidden");
    } else if (viewName === "running") {
        viewRunning.classList.remove("hidden");
    } else if (viewName === "result") {
        viewResult.classList.remove("hidden");
    }
}

function setTimelineState(nodeName, state) {
    const nodeEl = nodes[nodeName];
    if (!nodeEl) return;

    nodeEl.classList.remove("active", "completed", "idle");
    nodeEl.classList.add(state);

    const statusEl = nodeEl.querySelector(".node-status");
    if (statusEl) {
        statusEl.textContent = state;
    }
}

function resetTimeline() {
    Object.keys(nodes).forEach(node => {
        setTimelineState(node, "idle");
    });
}

function appendLog(category, message) {
    const line = document.createElement("div");
    line.className = `console-line ${category}-msg`;
    line.textContent = message;
    consoleLogs.appendChild(line);
    
    // Auto-scroll console
    consoleLogs.scrollTop = consoleLogs.scrollHeight;
}

// Clock Timer counter
function startTimer() {
    let seconds = 0;
    runningTimer.textContent = "00:00";
    
    if (timerInterval) clearInterval(timerInterval);
    
    timerInterval = setInterval(() => {
        seconds++;
        const mins = String(Math.floor(seconds / 60)).padStart(2, "0");
        const secs = String(seconds % 60).padStart(2, "0");
        runningTimer.textContent = `${mins}:${secs}`;
    }, 1000);
}

function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}

// Toast alerts system
function showToast(type, message) {
    toast.className = `toast-notification ${type}`;
    toastMessage.textContent = message;

    if (type === "success") {
        toastIcon.className = "fa-solid fa-circle-check";
    } else if (type === "error") {
        toastIcon.className = "fa-solid fa-circle-exclamation";
    } else {
        toastIcon.className = "fa-solid fa-circle-info";
    }

    toast.classList.remove("hidden");

    setTimeout(() => {
        toast.classList.add("hidden");
    }, 4000);
}
