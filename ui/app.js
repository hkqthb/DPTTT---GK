document.addEventListener("DOMContentLoaded", () => {
    // === CÁC BIẾN TRẠNG THÁI ===
    let activeTab = "visualizer";
    let activeHead = "mean"; // 'mean' hoặc index 0, 1, 2, 3...
    let attentionData = null; // Lưu trữ dữ liệu analyze hiện tại
    let currentGenerateText = "";
    let isGenerating = false;
    let generateInterval = null;
    let candidatesChart = null;
    let benchmarkChart = null;

    // === DOM ELEMENTS ===
    const navItems = document.querySelectorAll(".nav-item");
    const tabContents = document.querySelectorAll(".tab-content");
    const tabTitle = document.getElementById("current-tab-title");
    const tabDesc = document.getElementById("current-tab-desc");
    const modelBadgeInfo = document.getElementById("badge-info");
    const themeToggleBtn = document.getElementById("btn-theme-toggle");

    // === 0. LOGIC CHUYỂN ĐỔI THEME (LIGHT / DARK) ===
    // Thiết lập Light Mode làm mặc định như yêu cầu của người dùng, hoặc lấy từ localStorage nếu có
    const savedTheme = localStorage.getItem("theme") || "light";
    if (savedTheme === "light") {
        document.body.classList.add("light-theme");
        updateThemeIcon(true);
    } else {
        document.body.classList.remove("light-theme");
        updateThemeIcon(false);
    }

    themeToggleBtn.addEventListener("click", () => {
        const isLight = document.body.classList.toggle("light-theme");
        localStorage.setItem("theme", isLight ? "light" : "dark");
        updateThemeIcon(isLight);
        
        // Cập nhật lại màu sắc biểu đồ nếu biểu đồ đang hiển thị
        if (candidatesChart) {
            updateChartTheme(candidatesChart);
        }
        if (benchmarkChart) {
            updateChartTheme(benchmarkChart);
        }
    });

    function updateThemeIcon(isLight) {
        const icon = themeToggleBtn.querySelector("i");
        if (isLight) {
            icon.className = "fa-solid fa-moon";
            themeToggleBtn.title = "Chuyển sang Giao diện Tối";
        } else {
            icon.className = "fa-solid fa-sun";
            themeToggleBtn.title = "Chuyển sang Giao diện Sáng";
        }
    }

    // Lấy màu sắc chữ dựa vào theme hiện tại để vẽ Chart.js
    function getThemeChartColors() {
        const isLight = document.body.classList.contains("light-theme");
        return {
            textColor: isLight ? "#0f172a" : "#f3f4f6",
            gridColor: isLight ? "rgba(0, 0, 0, 0.05)" : "rgba(255, 255, 255, 0.05)",
            // Chart.js canvas KHÔNG hỗ trợ var() CSS variables, phải dùng trực tiếp hex/rgba
            secondaryColor: isLight ? "#0891b2" : "#06b6d4",
            secondaryBg: isLight ? "rgba(8, 145, 178, 0.15)" : "rgba(6, 182, 212, 0.1)",
            accentColor: isLight ? "#e11d48" : "#f43f5e",
            accentBg: isLight ? "rgba(225, 29, 72, 0.1)" : "rgba(244, 63, 94, 0.1)",
            primaryColor: isLight ? "#6d28d9" : "#8b5cf6"
        };
    }

    // Cập nhật theme động cho các chart đang có mà không cần tạo mới hoàn toàn
    function updateChartTheme(chart) {
        const colors = getThemeChartColors();
        
        if (chart.options.scales.x) {
            chart.options.scales.x.ticks.color = colors.textColor;
            chart.options.scales.x.grid.color = colors.gridColor;
            if (chart.options.scales.x.title) {
                chart.options.scales.x.title.color = colors.textColor;
            }
        }
        if (chart.options.scales.y) {
            chart.options.scales.y.ticks.color = colors.textColor;
            chart.options.scales.y.grid.color = colors.gridColor;
            if (chart.options.scales.y.title) {
                chart.options.scales.y.title.color = colors.textColor;
            }
        }
        if (chart.options.plugins.legend && chart.options.plugins.legend.labels) {
            chart.options.plugins.legend.labels.color = colors.textColor;
        }
        
        // Cập nhật dataset colors nếu là benchmark chart (2 datasets)
        if (chart.data.datasets.length === 2) {
            chart.data.datasets[0].borderColor = colors.secondaryColor;
            chart.data.datasets[0].backgroundColor = colors.secondaryBg;
            chart.data.datasets[1].borderColor = colors.accentColor;
            chart.data.datasets[1].backgroundColor = colors.accentBg;
        }
        // Cập nhật dataset colors nếu là candidates chart (1 dataset)
        if (chart.data.datasets.length === 1) {
            chart.data.datasets[0].backgroundColor = colors.secondaryBg;
            chart.data.datasets[0].borderColor = colors.secondaryColor;
        }
        
        chart.update();
    }

    // === 1. ĐIỀU HƯỚNG TAB ===
    navItems.forEach(item => {
        item.addEventListener("click", () => {
            const targetTab = item.getAttribute("data-tab");
            switchTab(targetTab);
        });
    });

    function switchTab(tabName) {
        activeTab = tabName;
        
        // Update menu active class
        navItems.forEach(item => {
            if (item.getAttribute("data-tab") === tabName) {
                item.classList.add("active");
            } else {
                item.classList.remove("active");
            }
        });

        // Show/hide content
        tabContents.forEach(content => {
            if (content.id === `tab-${tabName}`) {
                content.classList.add("active");
            } else {
                content.classList.remove("active");
            }
        });

        // Update titles
        const titles = {
            visualizer: { title: "Visualizer Attention", desc: "Trực quan hóa trọng số Self-Attention của các head trên chuỗi ký tự." },
            generator: { title: "Autoregressive Text Generator", desc: "Mô phỏng sinh từ tự hồi quy từng bước, kết hợp Attention và N-gram." },
            benchmark: { title: "Performance Benchmark", desc: "So sánh tốc độ tính toán giữa bản Vectorized (dùng ma trận) và Naive (dùng vòng lặp)." },
            config: { title: "Model Configuration", desc: "Khởi tạo lại mô hình với kích thước embedding (d_model) và số lượng heads mới." }
        };

        tabTitle.textContent = titles[tabName].title;
        tabDesc.textContent = titles[tabName].desc;

        // Xử lý dừng generator nếu thoát khỏi tab generator
        if (tabName !== "generator") {
            stopAutoGeneration();
        }
    }

    // === 2. TAB 1: VISUALIZER ATTENTION ===
    const visualizerInput = document.getElementById("visualizer-input");
    const btnAnalyze = document.getElementById("btn-analyze");
    const heatmapGrid = document.getElementById("heatmap-grid");
    const headTabsContainer = document.getElementById("head-tabs");
    const promptButtons = document.querySelectorAll(".prompt-btn");

    // Chọn prompt mẫu nhanh
    promptButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            visualizerInput.value = btn.textContent;
            analyzeAttention();
        });
    });

    btnAnalyze.addEventListener("click", analyzeAttention);

    async function analyzeAttention() {
        const text = visualizerInput.value.trim();
        if (!text) return;

        btnAnalyze.disabled = true;
        btnAnalyze.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang tính toán...';

        try {
            const response = await fetch("/api/attention", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text })
            });

            if (!response.ok) throw new Error("API error");

            attentionData = await response.json();
            
            // Khởi tạo các button chọn Head dựa trên số lượng head thực tế nhận được
            setupHeadSelectors(attentionData.attention_weights.length);
            
            // Vẽ heatmap
            renderHeatmap();
        } catch (error) {
            console.error(error);
            heatmapGrid.innerHTML = `
                <div class="heatmap-placeholder">
                    <i class="fa-solid fa-triangle-exclamation" style="color:var(--accent);font-size:3rem;"></i>
                    <p>Có lỗi xảy ra khi phân tích dữ liệu.</p>
                </div>`;
        } finally {
            btnAnalyze.disabled = false;
            btnAnalyze.innerHTML = '<i class="fa-solid fa-circle-play"></i> Phân Tích Attention';
        }
    }

    function setupHeadSelectors(numHeads) {
        let html = `<button class="head-btn ${activeHead === "mean" ? "active" : ""}" data-head="mean">Average</button>`;
        for (let i = 0; i < numHeads; i++) {
            html += `<button class="head-btn ${activeHead == i ? "active" : ""}" data-head="${i}">Head ${i}</button>`;
        }
        headTabsContainer.innerHTML = html;

        // Thêm listener
        headTabsContainer.querySelectorAll(".head-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                headTabsContainer.querySelectorAll(".head-btn").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                activeHead = btn.getAttribute("data-head");
                renderHeatmap();
            });
        });
    }

    function renderHeatmap() {
        if (!attentionData || !attentionData.tokens.length) return;

        const tokens = attentionData.tokens;
        const weights = attentionData.attention_weights; // shape: (num_heads, seq_len, seq_len)
        const seqLen = tokens.length;

        // Tính ma trận hiển thị tùy theo head được chọn
        let displayMatrix = [];
        if (activeHead === "mean") {
            const numHeads = weights.length;
            for (let i = 0; i < seqLen; i++) {
                displayMatrix.push([]);
                for (let j = 0; j < seqLen; j++) {
                    let sum = 0;
                    for (let h = 0; h < numHeads; h++) {
                        sum += weights[h][i][j];
                    }
                    displayMatrix[i].push(sum / numHeads);
                }
            }
        } else {
            const headIdx = parseInt(activeHead);
            displayMatrix = weights[headIdx];
        }

        // Tạo Grid style
        // Cần thêm 1 hàng trên cùng cho header và 1 cột bên trái cho y-labels
        heatmapGrid.style.gridTemplateColumns = `100px repeat(${seqLen}, 45px)`;
        heatmapGrid.innerHTML = "";

        // Ô trống trên cùng bên trái
        const emptyCell = document.createElement("div");
        emptyCell.style.width = "100px";
        emptyCell.style.height = "35px";
        heatmapGrid.appendChild(emptyCell);

        // Header chứa các từ khóa cột (Keys)
        for (let j = 0; j < seqLen; j++) {
            const colHeader = document.createElement("div");
            colHeader.className = "heatmap-col-label";
            colHeader.textContent = tokens[j];
            colHeader.style.cssText = "font-weight:600; font-size:0.75rem; text-align:center; transform: rotate(-30deg); transform-origin: bottom left; height: 35px; white-space: nowrap; color: var(--text-secondary);";
            heatmapGrid.appendChild(colHeader);
        }

        // Từng dòng (Queries)
        for (let i = 0; i < seqLen; i++) {
            // Label dòng bên trái
            const rowLabel = document.createElement("div");
            rowLabel.className = "heatmap-row-label";
            rowLabel.textContent = tokens[i];
            rowLabel.style.cssText = "font-weight:600; font-size:0.75rem; display:flex; align-items:center; justify-content:flex-end; padding-right:10px; height:45px; color: var(--text-secondary);";
            heatmapGrid.appendChild(rowLabel);

            // Các ô weights trong dòng
            for (let j = 0; j < seqLen; j++) {
                const val = displayMatrix[i][j];
                const cell = document.createElement("div");
                cell.className = "heatmap-cell";
                cell.style.width = "45px";
                cell.style.height = "45px";
                
                // Causal mask che đi tương lai (các ô có giá trị gần 0 do masking)
                // Đặt opacity nền tỉ lệ thuận với weight
                const isLight = document.body.classList.contains("light-theme");
                const r = isLight ? 109 : 139;
                const g = isLight ? 40 : 92;
                const b = isLight ? 217 : 246;
                cell.style.backgroundColor = `rgba(${r}, ${g}, ${b}, ${val})`;
                
                // Đặt màu text trắng hoặc đen/xám sáng tùy độ sáng nền & theme
                cell.style.color = val > 0.4 ? "#fff" : (isLight ? "rgba(0,0,0,0.6)" : "rgba(255,255,255,0.5)");
                cell.textContent = val > 0.05 ? val.toFixed(2) : "";

                // Tooltip
                cell.setAttribute("data-tooltip", `Từ "${tokens[i]}" chú ý đến "${tokens[j]}": ${val.toFixed(4)}`);

                // Highlight dòng và cột khi hover
                cell.addEventListener("mouseenter", () => {
                    highlightGrid(i, j, seqLen);
                });
                cell.addEventListener("mouseleave", () => {
                    clearGridHighlight();
                });

                heatmapGrid.appendChild(cell);
            }
        }
    }

    function highlightGrid(rowIndex, colIndex, seqLen) {
        const cells = heatmapGrid.querySelectorAll(".heatmap-cell");
        cells.forEach((cell, idx) => {
            const r = Math.floor(idx / seqLen);
            const c = idx % seqLen;
            if (r === rowIndex || c === colIndex) {
                cell.classList.add("active-row-col");
            } else {
                cell.classList.remove("active-row-col");
            }
        });
    }

    function clearGridHighlight() {
        const cells = heatmapGrid.querySelectorAll(".heatmap-cell");
        cells.forEach(cell => cell.classList.remove("active-row-col"));
    }

    // === 3. TAB 2: TEXT GENERATOR ===
    const genSeedInput = document.getElementById("generator-seed");
    const genBlendSlider = document.getElementById("generator-blend");
    const genBlendVal = document.getElementById("blend-val");
    const genTempSlider = document.getElementById("generator-temp");
    const genTempVal = document.getElementById("temp-val");
    const genOutputContainer = document.getElementById("generator-output-text");
    const highlightTokensContainer = document.getElementById("highlight-tokens-container");

    const btnGenStep = document.getElementById("btn-gen-step");
    const btnGenAuto = document.getElementById("btn-gen-auto");
    const btnGenReset = document.getElementById("btn-gen-reset");
    const seedButtons = document.querySelectorAll(".seed-btn");

    genBlendSlider.addEventListener("input", (e) => {
        genBlendVal.textContent = parseFloat(e.target.value).toFixed(2);
    });
    genTempSlider.addEventListener("input", (e) => {
        genTempVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    seedButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            genSeedInput.value = btn.textContent;
            resetGenerator();
        });
    });

    btnGenStep.addEventListener("click", () => generateStep());
    btnGenAuto.addEventListener("click", toggleAutoGeneration);
    btnGenReset.addEventListener("click", resetGenerator);

    function resetGenerator() {
        stopAutoGeneration();
        currentGenerateText = genSeedInput.value.trim();
        genOutputContainer.innerHTML = `<span class="seed-part">${currentGenerateText}</span>`;
        highlightTokensContainer.innerHTML = `<p class="placeholder-text">Bắt đầu sinh từ để xem Attention Highlight</p>`;
        
        if (candidatesChart) {
            candidatesChart.destroy();
            candidatesChart = null;
        }

        btnGenStep.disabled = false;
        btnGenAuto.disabled = false;
    }

    async function generateStep() {
        if (!currentGenerateText) {
            currentGenerateText = genSeedInput.value.trim();
        }

        btnGenStep.disabled = true;
        let reachedEos = false;

        try {
            const response = await fetch("/api/generate_step", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    current_text: currentGenerateText,
                    temperature: parseFloat(genTempSlider.value),
                    blend_ratio: parseFloat(genBlendSlider.value)
                })
            });

            if (!response.ok) throw new Error("Gen step error");

            const data = await response.json();
            
            // Thêm token mới sinh vào box output
            const isWord = data.new_text.includes(" ");
            const delimiter = isWord ? " " : "";
            
            // Cập nhật text hiện tại
            currentGenerateText = data.new_text;
            
            // Tránh trình duyệt hiểu nhầm <EOS> là thẻ HTML
            const displayToken = data.next_token.replace(/</g, "&lt;").replace(/>/g, "&gt;");

            // Update Output DOM
            if (data.next_token === "<EOS>") {
                genOutputContainer.innerHTML += `${delimiter}<span class="gen-part" style="color: var(--accent); font-weight: 700; background: rgba(244, 63, 94, 0.15); padding: 2px 6px; border-radius: 4px; border: 1px solid var(--accent); margin: 0 4px;">${displayToken}</span>`;
            } else {
                genOutputContainer.innerHTML += `${delimiter}<span class="gen-part" style="color: var(--primary); font-weight: 500;">${displayToken}</span>`;
            }

            // Vẽ biểu đồ candidates
            renderCandidatesChart(data.top_candidates);

            // Vẽ Attention Highlight
            renderAttentionHighlight(data.tokens, data.attention_weights);

            // Dừng sinh tự động nếu gặp EOS
            if (data.next_token === "<EOS>") {
                stopAutoGeneration();
                reachedEos = true;
                btnGenAuto.disabled = true;
                return false;
            }

            return true;
        } catch (error) {
            console.error(error);
            stopAutoGeneration();
            return false;
        } finally {
            if (reachedEos) {
                btnGenStep.disabled = true;
            } else {
                btnGenStep.disabled = false;
            }
        }
    }

    function toggleAutoGeneration() {
        if (isGenerating) {
            stopAutoGeneration();
        } else {
            startAutoGeneration();
        }
    }

    function startAutoGeneration() {
        isGenerating = true;
        btnGenAuto.innerHTML = '<i class="fa-solid fa-pause"></i> Tạm Dừng';
        btnGenAuto.classList.remove("btn-primary");
        btnGenAuto.classList.add("btn-secondary");
        
        generateInterval = setInterval(async () => {
            const success = await generateStep();
            if (!success) {
                stopAutoGeneration();
            }
        }, 1000); // 1 giây sinh 1 từ
    }

    function stopAutoGeneration() {
        isGenerating = false;
        if (generateInterval) {
            clearInterval(generateInterval);
            generateInterval = null;
        }
        btnGenAuto.innerHTML = '<i class="fa-solid fa-play"></i> Sinh Tự Động';
        btnGenAuto.classList.remove("btn-secondary");
        btnGenAuto.classList.add("btn-primary");
    }

    function renderCandidatesChart(candidates) {
        const labels = candidates.map(c => c.token);
        const probs = candidates.map(c => c.prob);
        const colors = getThemeChartColors();

        const ctx = document.getElementById("candidatesChart").getContext("2d");
        
        if (candidatesChart) {
            candidatesChart.data.labels = labels;
            candidatesChart.data.datasets[0].data = probs;
            updateChartTheme(candidatesChart);
        } else {
            candidatesChart = new Chart(ctx, {
                type: "bar",
                data: {
                    labels: labels,
                    datasets: [{
                        label: "Xác suất kết hợp (p_final)",
                        data: probs,
                        backgroundColor: colors.secondaryBg,
                        borderColor: colors.secondaryColor,
                        borderWidth: 1,
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: "y", // Cột nằm ngang
                    scales: {
                        x: {
                            max: 1.0,
                            grid: { color: colors.gridColor },
                            ticks: { color: colors.textColor }
                        },
                        y: {
                            grid: { display: false },
                            ticks: { color: colors.textColor }
                        }
                    },
                    plugins: {
                        legend: { display: false }
                    }
                }
            });
        }
    }

    function renderAttentionHighlight(tokens, weights) {
        // Lấy trọng số của token cuối cùng chiếu đến các token trước đó
        // weights shape: (num_heads, seq_len, seq_len)
        const seqLen = tokens.length;
        if (seqLen === 0) return;

        // Tính trung bình cộng attention weights qua tất cả các heads cho token cuối cùng
        const numHeads = weights.length;
        let lastTokenAttn = [];
        
        for (let j = 0; j < seqLen; j++) {
            let sum = 0;
            for (let h = 0; h < numHeads; h++) {
                sum += weights[h][seqLen - 1][j]; // weights của token cuối (seqLen-1) đối với token j
            }
            lastTokenAttn.push(sum / numHeads);
        }

        highlightTokensContainer.innerHTML = "";
        
        tokens.forEach((token, idx) => {
            const attnVal = lastTokenAttn[idx] || 0;
            const span = document.createElement("span");
            span.className = "highlight-token";
            span.textContent = token;
            
            // Background color alpha tỷ lệ thuận với weight
            const isLight = document.body.classList.contains("light-theme");
            const r = isLight ? 109 : 139;
            const g = isLight ? 40 : 92;
            const b = isLight ? 217 : 246;
            span.style.backgroundColor = `rgba(${r}, ${g}, ${b}, ${attnVal * 0.85})`;
            span.style.borderColor = `rgba(${r}, ${g}, ${b}, ${attnVal})`;
            span.title = `Attention Weight: ${attnVal.toFixed(4)}`;

            // Đánh dấu token nguồn
            if (idx === seqLen - 1) {
                span.classList.add("source");
            }
            
            highlightTokensContainer.appendChild(span);
        });
    }

    // === 4. TAB 3: PERFORMANCE BENCHMARK ===
    const btnRunBenchmark = document.getElementById("btn-run-benchmark");
    const benchmarkTableBody = document.querySelector("#benchmark-table tbody");

    btnRunBenchmark.addEventListener("click", runBenchmark);

    async function runBenchmark() {
        btnRunBenchmark.disabled = true;
        btnRunBenchmark.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang đo hiệu năng...';
        
        benchmarkTableBody.innerHTML = `
            <tr>
                <td colspan="4" style="text-align:center; padding: 2rem;">
                    <i class="fa-solid fa-spinner fa-spin" style="font-size:2rem; color:var(--primary); margin-bottom:1rem; display:block;"></i>
                    Đang chạy pipeline Attention trên CPU. Vui lòng đợi trong giây lát...
                </td>
            </tr>`;

        try {
            const seqLengths = [10, 30, 60, 100, 150]; // Giới hạn seq_len vừa phải tránh loop Python quá lâu
            const response = await fetch("/api/benchmark", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ seq_lengths: seqLengths })
            });

            if (!response.ok) throw new Error("Benchmark error");

            const data = await response.json();
            
            // Vẽ bảng kết quả
            renderBenchmarkTable(data.results);

            // Vẽ đồ thị so sánh
            renderBenchmarkChart(data.results);
        } catch (error) {
            console.error(error);
            benchmarkTableBody.innerHTML = `
                <tr>
                    <td colspan="4" style="text-align:center; color: var(--accent);">
                        Có lỗi xảy ra trong quá trình benchmark.
                    </td>
                </tr>`;
        } finally {
            btnRunBenchmark.disabled = false;
            btnRunBenchmark.innerHTML = '<i class="fa-solid fa-bolt"></i> Bắt đầu Benchmark';
        }
    }

    function renderBenchmarkTable(results) {
        benchmarkTableBody.innerHTML = "";
        results.forEach(row => {
            const tr = document.createElement("tr");
            
            const naiveStr = row.naive_time_ms !== null ? `${row.naive_time_ms.toFixed(2)} ms` : "N/A (Quá lâu)";
            
            let speedupStr = "N/A";
            if (row.naive_time_ms !== null && row.vectorized_time_ms > 0) {
                const factor = row.naive_time_ms / row.vectorized_time_ms;
                speedupStr = `<span class="speedup-badge">${factor.toFixed(1)}x Faster</span>`;
            }

            tr.innerHTML = `
                <td><strong>L = ${row.seq_len}</strong></td>
                <td style="color:var(--secondary); font-weight:600;">${row.vectorized_time_ms.toFixed(2)} ms</td>
                <td style="color:var(--accent);">${naiveStr}</td>
                <td>${speedupStr}</td>
            `;
            benchmarkTableBody.appendChild(tr);
        });
    }

    function renderBenchmarkChart(results) {
        const labels = results.map(r => `L=${r.seq_len}`);
        const vecData = results.map(r => r.vectorized_time_ms);
        const naiveData = results.map(r => r.naive_time_ms);
        const colors = getThemeChartColors();

        const ctx = document.getElementById("benchmarkChart").getContext("2d");

        if (benchmarkChart) {
            benchmarkChart.destroy();
        }

        benchmarkChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "Vectorized NumPy (Ma trận)",
                        data: vecData,
                        borderColor: colors.secondaryColor,
                        backgroundColor: colors.secondaryBg,
                        borderWidth: 2,
                        tension: 0.3,
                        fill: true,
                        pointRadius: 5,
                        pointHoverRadius: 7
                    },
                    {
                        label: "Naive Loops (Vòng lặp)",
                        data: naiveData,
                        borderColor: colors.accentColor,
                        backgroundColor: colors.accentBg,
                        borderWidth: 2,
                        tension: 0.3,
                        fill: true,
                        pointRadius: 5,
                        pointHoverRadius: 7
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        type: "logarithmic",
                        title: { display: true, text: "Thời gian chạy (ms) - Thang Log", color: colors.textColor },
                        grid: { color: colors.gridColor },
                        ticks: { 
                            color: colors.textColor,
                            callback: function(value, index, values) {
                                // Hiển thị các giá trị chẵn trên thang log
                                if (value === 0.1 || value === 0.2 || value === 0.5 || 
                                    value === 1 || value === 2 || value === 5 || 
                                    value === 10 || value === 20 || value === 50 || 
                                    value === 100 || value === 200 || value === 500 || value === 1000) {
                                    return value + " ms";
                                }
                                return null;
                            }
                        },
                        min: 0.1 // Tránh log(0)
                    },
                    x: {
                        title: { display: true, text: "Chiều dài chuỗi (Sequence Length)", color: colors.textColor },
                        grid: { display: false },
                        ticks: { color: colors.textColor }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: colors.textColor }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const value = context.parsed.y;
                                return context.dataset.label + ': ' + (value !== null ? value.toFixed(3) + ' ms' : 'N/A');
                            }
                        }
                    }
                }
            }
        });
    }

    // === 5. TAB 4: CONFIGURATION ===
    const cfgDmodel = document.getElementById("cfg-dmodel");
    const cfgHeads = document.getElementById("cfg-heads");
    const btnReinitModel = document.getElementById("btn-reinit-model");

    btnReinitModel.addEventListener("click", reinitModel);

    async function reinitModel() {
        const d_model = parseInt(cfgDmodel.value);
        const num_heads = parseInt(cfgHeads.value);

        if (d_model % num_heads !== 0) {
            alert(`Lỗi cấu hình: embedding dimension (${d_model}) phải chia hết cho số Heads (${num_heads})!`);
            return;
        }

        btnReinitModel.disabled = true;
        btnReinitModel.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang khởi tạo mô hình...';

        try {
            const response = await fetch("/api/init", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ d_model, num_heads })
            });

            if (!response.ok) throw new Error("Reinit model error");
            const data = await response.json();

            // Cập nhật badge hiển thị
            modelBadgeInfo.textContent = `Active: d_model=${d_model} | Heads=${num_heads}`;
            
            // Xóa dữ liệu cũ
            attentionData = null;
            heatmapGrid.innerHTML = `
                <div class="heatmap-placeholder">
                    <i class="fa-solid fa-check-double placeholder-icon" style="color:var(--secondary);"></i>
                    <p>Mô hình đã khởi tạo lại thành công!<br>Nhập câu mới và chọn "Phân Tích Attention" để xem kết quả.</p>
                </div>`;
            
            resetGenerator();

            alert(`Khởi tạo thành công! Từ điển có ${data.vocab_size} từ.`);
            switchTab("visualizer");
        } catch (error) {
            console.error(error);
            alert("Có lỗi xảy ra khi khởi tạo mô hình.");
        } finally {
            btnReinitModel.disabled = false;
            btnReinitModel.innerHTML = '<i class="fa-solid fa-wrench"></i> Khởi tạo & Xây dựng lại mô hình';
        }
    }

    // === KHỞI TẠO BAN ĐẦU ===
    // Chạy mặc định phân tích câu mẫu đầu tiên
    analyzeAttention();
});
